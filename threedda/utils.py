import os
import wandb
import torch
from argparse import Namespace
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
# sys.path.append("/home/memmelma/Projects/3dda/3d_diffuser_actor")
sys.path.append("/home/memmelma/Projects/robotic/3d_diffuser_actor")
from diffuser_actor import DiffuserActor, DiffuserJointer

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

def prepare_batch(sample, clip_embedder, history, horizon, obs_crop=False, obs_noise_std=0.0, obs_path=False, obs_mask=False, device=None):
    # gt_trajectory: (B, trajectory_length, 3+4+X)
    # trajectory_mask: (B, trajectory_length)
    # timestep: (B, 1)
    # rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
    # pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
    # instruction: (B, max_instruction_length, 512)
    # curr_gripper: (B, nhist, 3+4+X)

    qpos = sample["obs"]["qpos"]
    # discrete gripper state for action prediction -> BCE loss
    gripper_state_discrete = sample["obs"]["gripper_state_discrete"].float()
    # future actions
    gt_trajectory = torch.cat((qpos[:, history:], gripper_state_discrete[:, history:]), dim=-1)
    # past actions
    curr_gripper = torch.cat((qpos[:, :history], gripper_state_discrete[:, :history]), dim=-1)
    # (optional) add noise to qpos obs
    if obs_noise_std > 0:
        curr_gripper = curr_gripper + torch.normal(0, obs_noise_std, curr_gripper.shape)
    
    for k in sample["obs"].keys():
        sample["obs"][k] = sample["obs"][k][:, history-1]

    # # # HACK
    # path = True
    # if path:
    #     from utils.paths import generate_path_2d_from_obs, add_path_2d_to_img, smooth_path_rdp
    #     # pass   
    # #     paths = []
    # #     for i in range(len(sample["obs"]["rgb"])):
    # #         obs_np = {
    # #             "ee_pos": sample["obs"]["ee_pos"][i].cpu().numpy(),
    # #             "camera_intrinsic": sample["obs"]["camera_intrinsic"][i].cpu().numpy(),
    # #             "camera_extrinsic": sample["obs"]["camera_extrinsic"][i].cpu().numpy(),
    # #         }
    # #         path = generate_path_2d_from_obs(obs_np)
    # #         # TODO fix the tolerance to account for unscaled path
    # #         path_smooth = smooth_path_rdp(path, tolerance=8)
    # #         paths.append(path_smooth)
        
    #     for i in range(len(sample["obs"]["path"])):
    #         img = sample["obs"]["rgb"][i].cpu().numpy()
    #         path = sample["obs"]["path"][i].cpu().numpy()
    #         # unpad path, removing [0,0] values

    #         sample["obs"]["rgb"][i] = torch.from_numpy(add_path_2d_to_img(img, path)).to(device)

            # import matplotlib.pyplot as plt
            # plt.imsave(f"path_img.png", img)
    
    img_key = "rgb" if not obs_path else "path_rgb"
    depth_key = "depth" if not obs_mask else "mask_depth"
    
    B, H, W = sample["obs"][depth_key].shape
    points = depth_to_points_torch_batched(sample["obs"][depth_key].reshape(B, H, W), sample["obs"]["camera_intrinsic"].reshape(B, 3, 3), sample["obs"]["camera_extrinsic"].reshape(B, 4, 4), depth_scale=1000.)
    colors = sample["obs"][img_key].reshape(B, H * W, 3)

    if obs_crop:
        from utils.pointclouds import zero_points
        points, colors = zero_points(points, colors, crop_min=[0.0, -0.5, 0.01], crop_max=[0.8, 0.5, 1.])
    
    points = points.reshape(B, H, W, 3)
    colors = colors.reshape(B, H, W, 3)
    pcd_obs = points.permute(0, 3, 1, 2).unsqueeze(1).float()
    rgb_obs = colors.permute(0, 3, 1, 2).unsqueeze(1).float() / 255.0

    instructions = ["pick the red cube"] * B
    instruction = torch.stack([clip_embedder.embed_instruction(instr) for instr in instructions])

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

def get_model(da_config):

    assert da_config.action_space in ["abs_ee", "joint"], "Invalid action space: {}".format(da_config.action_space)

    if da_config.action_space == "abs_ee":
        model = DiffuserActor(
            backbone="clip",
            image_size=(256,256),
            embedding_dim=da_config.embedding_dim,
            num_vis_ins_attn_layers=2,
            use_instruction=True,
            fps_subsampling_factor=da_config.fps_subsampling_factor,
            # TODO: check those
            # gripper_loc_bounds=np.array([[-1.0, -1.0, 0], [1.0, 1.0, 1.0]]),
            gripper_loc_bounds=np.array([
                [0, -0.5, -0.01],
                [1, 0.5, 0.5]
            ]),
            # rotation_parametrization="quat", # -> 6D is hardcoded ...
            rotation_parametrization='6D',
            quaternion_format="xyzw",
            diffusion_timesteps=da_config.diffusion_timesteps,
            nhist=da_config.history,
            relative=False,
            lang_enhanced=False,
            loss_weights=[30., 10., 1.]
            # loss_weights=[30., 10., 0.]
        )
    elif da_config.action_space == "joint":
        model = DiffuserJointer(
            backbone="clip",
            image_size=da_config.image_size,
            embedding_dim=da_config.embedding_dim,
            num_attn_heads=da_config.num_attn_heads,
            num_vis_ins_attn_layers=2,
            use_instruction=True,
            fps_subsampling_factor=5,
            # TODO: check those
            gripper_loc_bounds=da_config.gripper_loc_bounds, # np.array([[-1.0, -1.0, 0], [1.0, 1.0, 1.0]]),
            joint_loc_bounds=da_config.joint_loc_bounds,
            loss_weights=da_config.loss_weights,
            diffusion_timesteps=da_config.diffusion_timesteps,
            nhist=da_config.history,
            relative=False,
            lang_enhanced=False
        )

    return model

def get_optimizer(model, lr=1e-4):
    """Initialize optimizer."""
    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0, "lr": lr},
        {"params": [], "weight_decay": 5e-4, "lr": lr}
    ]
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)
        else:
            optimizer_grouped_parameters[1]["params"].append(param)
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    return optimizer

def load_checkpoint(checkpoint):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(checkpoint))

    ckpt_dict = torch.load(checkpoint, map_location="cpu")

    model_config = ckpt_dict["model_config"]
    model = get_model(model_config)
    model.load_state_dict(ckpt_dict["model_state_dict"])

    optimizer = get_optimizer(model, lr=model_config.lr)
    if 'optimizer_state_dict' in ckpt_dict:
        optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
        for p in range(len(optimizer.param_groups)):
            optimizer.param_groups[p]['lr'] = model_config.lr
    
    start_iter = ckpt_dict.get("iter", 0)
    best_loss = ckpt_dict.get("best_loss", None)

    wandb_dict = {
        "run_id": ckpt_dict.get("wandb_run_id", None),
        "entity": ckpt_dict.get("wandb_entity", None),
        "project": ckpt_dict.get("wandb_project", None),
    }

    print("=> loaded successfully '{}' (step {})".format(
        checkpoint, ckpt_dict.get("iter", 0)
    ))
    del ckpt_dict
    torch.cuda.empty_cache()
    return model, optimizer, start_iter, best_loss, wandb_dict, model_config

def save_checkpoint(
        model, optimizer,
        iter, new_loss, best_loss,
        wandb_config, model_config,
        log_dir
    ):
    """Save checkpoint if requested."""
    
    new_best = False
    if new_loss is None or best_loss is None or new_loss <= best_loss:
        best_loss = new_loss
        new_best = True

    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        
        "iter": iter + 1,
        "best_loss": best_loss,
        
        "wandb_run_id": wandb_config["run_id"],
        "wandb_entity": wandb_config["entity"],
        "wandb_project": wandb_config["project"],

        "model_config": model_config
    }

    if new_best:
        torch.save(save_dict, os.path.join(log_dir, "best.pth"))
    if (iter + 1) % 50 == 0:
        torch.save(save_dict, os.path.join(log_dir, f"iter_{iter+1}.pth"))
    torch.save(save_dict, os.path.join(log_dir, "last.pth"))
    return best_loss

def plot_actions_and_log_wandb(pred_actions, true_actions, wandb_title, epoch):
    """
    Plots predicted vs. ground truth actions (7-dim) along with a corresponding image strip.
    Logs the plot to WandB.
    """

    ACTION_DIM_LABELS = [str(i) for i in range(pred_actions.shape[-1])]

    figure_layout = [ACTION_DIM_LABELS]
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([40, 5])

    # Ensure proper input formatting for actions
    pred_actions = np.array(pred_actions).squeeze()  # Bx7
    true_actions = np.array(true_actions).squeeze()  # Bx7

    # Plot actions for each dimension
    for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
        axs[action_label].plot(pred_actions[:, action_dim], label="Predicted")
        axs[action_label].plot(true_actions[:, action_dim], label="Ground Truth")
        axs[action_label].set_title(action_label)
        axs[action_label].set_xlabel("Time (steps)")
        axs[action_label].legend()

    plt.tight_layout()
    wandb.log({"epoch": epoch, wandb_title: wandb.Image(fig)})
    plt.close(fig)