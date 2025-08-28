import torch
from problem_reduction.vila.encode import scale_path
from problem_reduction.vila.decode import add_path_2d_to_img, add_mask_2d_to_img

def depth_to_points_torch_batched(depth, intrinsic, extrinsic, depth_scale=1000.0):
    B, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    depth = depth / depth_scale
    ylin = torch.arange(0, H, device=device, dtype=dtype)
    xlin = torch.arange(0, W, device=device, dtype=dtype)
    py, px = torch.meshgrid(ylin, xlin, indexing='ij')  # shape (H, W)
    px = px.unsqueeze(0).expand(B, -1, -1)
    py = py.unsqueeze(0).expand(B, -1, -1)

    fx, fy = intrinsic[:, 0, 0], intrinsic[:, 1, 1]
    cx, cy = intrinsic[:, 0, 2], intrinsic[:, 1, 2]

    px = (px - cx.view(-1, 1, 1)) * (depth / fx.view(-1, 1, 1))
    py = (py - cy.view(-1, 1, 1)) * (depth / fy.view(-1, 1, 1))
    ones = torch.ones_like(depth)

    points = torch.stack((px, py, depth, ones), dim=1)  # (B, 4, H, W)
    points = points.reshape(B, 4, -1)
    points = (extrinsic @ points).transpose(1, 2)[:, :, :3]  # (B, H*W, 3)
    return points

def prepare_batch(sample, history, horizon, obs_crop=False, obs_crop_cube=False, obs_noise_std=0.0, obs_path_mask_noise_std=0.0, obs_discrete_gripper=True, obs_no_proprio=False, obs_path=False, obs_mask=False, obs_mask_w_path=False, obs_gt=False, obs_hamster=False, obs_outlier=False, mask_pixels=10, action_space="joint", device=None):
    # gt_trajectory: (B, trajectory_length, 3+4+X)
    # trajectory_mask: (B, trajectory_length)
    # timestep: (B, 1)
    # rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
    # pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
    # instruction: (B, max_instruction_length, 512)
    # curr_gripper: (B, nhist, 3+4+X)

    # discrete gripper state for action prediction -> BCE loss
    gripper_state_discrete = sample["obs"]["gripper_state_discrete"].float()
    # continuous gripper state for observations -> normalize 0-1
    gripper_state_continuous = sample["obs"]["gripper_state_continuous"].float() / 0.04

    if action_space == "joint":
        act = sample["obs"]["qpos"]
    elif action_space == "abs_ee":
        act = sample["obs"]["ee_pose"]

    # future actions
    gt_trajectory = torch.cat((act[:, history:], gripper_state_discrete[:, history:]), dim=-1)
    # past actions
    if obs_discrete_gripper:
        curr_gripper = torch.cat((act[:, :history], gripper_state_discrete[:, :history]), dim=-1)
    else:
        curr_gripper = torch.cat((act[:, :history], gripper_state_continuous[:, :history]), dim=-1)
    # (optional) add noise to ee_pose/qpos obs
    if obs_noise_std > 0:
        curr_gripper = curr_gripper + torch.normal(0, obs_noise_std, curr_gripper.shape).to(curr_gripper.device)
    if obs_no_proprio:
        curr_gripper = torch.zeros_like(curr_gripper)

    for k in sample["obs"].keys():
        sample["obs"][k] = sample["obs"][k][:, history-1]

    img_key = "rgb"
    depth_key = "depth"
    tmp_device = sample["obs"][depth_key].device
    B, H, W = sample["obs"][depth_key].shape

    # store mask as 2d depth
    if obs_mask:
        mask_depths = []
        for mask, depth in zip(sample["obs"]["mask" if obs_gt else "mask_vlm"], sample["obs"][depth_key]):
            # unpad mask
            m = ~( (mask[:,0] == -1.) & (mask[:,1] == -1.) )
            mask_unpad = mask[m]
            mask_unpad = mask_unpad + torch.normal(0, obs_path_mask_noise_std, mask_unpad.shape).to(mask_unpad.device)
            mask_unpad = torch.clamp(mask_unpad, 0., 1.)
            mask_unpad = scale_path(mask_unpad, min_in=0., max_in=1., min_out=0., max_out=H)
            # add mask to depth
            mask_depth = add_mask_2d_to_img(depth.cpu().numpy(), mask_unpad.cpu().numpy(), mask_pixels=mask_pixels)
            mask_depths.append(torch.from_numpy(mask_depth))
        sample["obs"][depth_key] = torch.stack(mask_depths, dim=0).to(tmp_device)

    # store mask w/ pathas 2d depth
    if obs_mask_w_path:
        mask_depths = []
        for mask, path, depth in zip(sample["obs"]["mask" if obs_gt else "mask_vlm"], sample["obs"]["path" if obs_gt else "path_vlm"], sample["obs"][depth_key]):
            # unpad mask
            m = ~( (mask[:,0] == -1.) & (mask[:,1] == -1.) )
            mask_unpad = mask[m]
            mask_unpad = mask_unpad + torch.normal(0, obs_path_mask_noise_std, mask_unpad.shape).to(mask_unpad.device)
            mask_unpad = torch.clamp(mask_unpad, 0., 1.)
            mask_unpad = scale_path(mask_unpad, min_in=0., max_in=1., min_out=0., max_out=H)
            # unpad path
            m = ~( (path[:,0] == -1.) & (path[:,1] == -1.) )
            path_unpad = path[m]
            path_unpad = path_unpad + torch.normal(0, obs_path_mask_noise_std, path_unpad.shape).to(path_unpad.device)
            path_unpad = torch.clamp(path_unpad, 0., 1.)
            path_unpad = scale_path(path_unpad, min_in=0., max_in=1., min_out=0., max_out=H)
            # combine mask and path
            mask_w_path_unpad = torch.cat((mask_unpad, path_unpad), dim=0)
            # add mask w/ path to depth
            mask_depth = add_mask_2d_to_img(depth.cpu().numpy(), mask_w_path_unpad.cpu().numpy(), mask_pixels=mask_pixels)
            mask_depths.append(torch.from_numpy(mask_depth))
        sample["obs"][depth_key] = torch.stack(mask_depths, dim=0).to(tmp_device)

    # mask out pixels
    if obs_path and (obs_mask or obs_mask_w_path):
        sample["obs"][img_key][sample["obs"][depth_key] == 0] = 0.
    
    # store path as 2d image
    # NOTE: do this after RGB is masked out!
    if obs_path:
        path_rgbs = []
        for path, rgb in zip(sample["obs"]["path" if obs_gt else "path_vlm"], sample["obs"][img_key]):
            # unpad path
            m = ~( (path[:,0] == -1.) & (path[:,1] == -1.) )
            path_unpad = path[m]
            path_unpad = path_unpad + torch.normal(0, obs_path_mask_noise_std, path_unpad.shape).to(path_unpad.device)
            path_unpad = torch.clamp(path_unpad, 0., 1.)
            path_unpad = scale_path(path_unpad, min_in=0., max_in=1., min_out=0., max_out=H)
            # add path to rgb
            path_rgb = add_path_2d_to_img(rgb.cpu().numpy(), path_unpad.cpu().numpy())
            path_rgbs.append(torch.from_numpy(path_rgb))
        sample["obs"][img_key] = torch.stack(path_rgbs, dim=0).to(tmp_device)

    if obs_hamster:
        path_rgbs = []
        for path, rgb in zip(sample["obs"]["path" if obs_gt else "path_vlm"], sample["obs"][img_key]):
            # unpad path
            m = ~( (path[:,0] == -1.) & (path[:,1] == -1.) & (path[:,2] == -1.) )
            path_unpad = path[m]
            assert obs_path_mask_noise_std == 0.01, "HAMSTER used 0.01 noise"
            path_unpad = path_unpad + torch.normal(0, obs_path_mask_noise_std, path_unpad.shape).to(path_unpad.device)
            path_unpad = torch.clamp(path_unpad, 0., 1.)
            
            from problem_reduction.vila.inference_hamster import draw_lines_on_image_cv
            path_rgb = draw_lines_on_image_cv(rgb.cpu().numpy(), path_unpad.cpu().numpy(), draw_action=True)
            path_rgbs.append(torch.from_numpy(path_rgb))
        sample["obs"][img_key] = torch.stack(path_rgbs, dim=0).to(tmp_device)
    # import matplotlib.pyplot as plt
    # plt.imsave("mask_depth_rgb.png", sample["obs"][img_key][0].cpu().numpy())
    # plt.imsave("mask_depth_depth.png", sample["obs"][depth_key][0].cpu().numpy())
    # import IPython; IPython.embed()
    
    points = depth_to_points_torch_batched(sample["obs"][depth_key].reshape(B, H, W), sample["obs"]["camera_intrinsic"].reshape(B, 3, 3), sample["obs"]["camera_extrinsic"].reshape(B, 4, 4), depth_scale=1000.)
    colors = sample["obs"][img_key].reshape(B, H * W, 3)
    
    def zero_points(points, colors=None, crop_min=[-1.0, -1.0, -0.2], crop_max=[1.0, 1.0, 1.0]):
        crop_min = torch.tensor(crop_min, device=points.device).view(1, 1, 3)
        crop_max = torch.tensor(crop_max, device=points.device).view(1, 1, 3)

        mask_min = (points > crop_min).all(dim=-1)
        mask_max = (points < crop_max).all(dim=-1)
        valid_mask = mask_min & mask_max

        points[~valid_mask] = 0.
        if colors is not None:
            colors[~valid_mask] = 0.
        return points, colors
    
    # WARNING: zero_points with min < -1. will also crop the mask predictions from the colors/rgb!
    if obs_crop:
        raise NotImplementedError("obs_crop not supported")

        # # no table surface
        # points, colors = zero_points(points, colors, crop_min=[0.0, -0.5, 0.01], crop_max=[0.8, 0.5, 1.])

        # # just the table
        # points, colors = zero_points(points, colors, crop_min=[0.3, -0.2, -0.1], crop_max=[0.7, 0.2, 0.3])

        # # full workspace + robot
        points, colors = zero_points(points, colors, crop_min=[0., -0.3, -0.1], crop_max=[0.7, 0.3, 0.8])
    
    if obs_crop_cube:
        raise NotImplementedError("obs_crop not supported")
        points, colors = zero_points(points, colors, crop_min=[0.2, -0.3, 0.02], crop_max=[0.7, 0.3, 0.3])
    
    if obs_outlier:
        print("WARNING only use during inference - OOM otherwise")
        def remove_outliers(points, colors, k=3, threshold=2e-2):
            H, W, _ = points.shape
            pts = points.reshape(-1, 3)
            cols = colors.reshape(-1, 3)
            diffs = pts.unsqueeze(1) - pts.unsqueeze(0)
            dists = torch.norm(diffs, dim=2)
            knn_dists, _ = torch.topk(dists, k+1, largest=False)
            mean_knn_dist = knn_dists[:, 1:].mean(dim=1)
            outliers = mean_knn_dist > threshold
            cleaned_pts = pts.clone()
            cleaned_cols = cols.clone()
            cleaned_pts[outliers] = 0
            cleaned_cols[outliers] = 0
            return cleaned_pts.reshape(H, W, 3), cleaned_cols.reshape(H, W, 3)

        points, colors = remove_outliers(points, colors,k=3, threshold=2e-2)

    points = points.reshape(B, H, W, 3)
    colors = colors.reshape(B, H, W, 3)
    pcd_obs = points.permute(0, 3, 1, 2).unsqueeze(1).float()
    rgb_obs = colors.permute(0, 3, 1, 2).unsqueeze(1).float() / 255.0

    instruction = sample["obs"]["lang_instr"][:, :, history-1]
    
    trajectory_mask = torch.full((B, horizon, gt_trajectory.shape[-1]), False).float()

    batch = {
        "gt_trajectory": gt_trajectory.float(),
        "curr_gripper": curr_gripper.float(),
        "rgb_obs": rgb_obs.float(),
        "pcd_obs": pcd_obs.float(),
        "instruction": instruction.float(),
        "trajectory_mask": trajectory_mask.float(),
    }
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch
