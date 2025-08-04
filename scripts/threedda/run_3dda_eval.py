import json
import h5py
import torch
import imageio
import os
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from problem_reduction.threedda.text_embed import CLIPTextEmbedder
from problem_reduction.threedda.data import prepare_batch
from problem_reduction.threedda.model import load_checkpoint
from problem_reduction.threedda.vis_utils import plot_actions

from problem_reduction.robot.wrappers.framestack import FrameStackWrapper
from problem_reduction.robot.robot_env import CubeEnv
from problem_reduction.utils.normalize import denormalize

from problem_reduction.vila.inference_helpers import vila_inference_api

def visualize_pointcloud(env, data_path, demo_idx=0):
    def depth_to_points(depth, intrinsic, extrinsic, depth_scale=1000.0):
        height, width = depth.shape[:2]
        depth = depth.squeeze() / depth_scale
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsic[0, 2]) * (depth / intrinsic[0, 0])
        py = (py - intrinsic[1, 2]) * (depth / intrinsic[1, 1])
        points = np.stack((px, py, depth, np.ones(depth.shape)), axis=-1)
        points = (extrinsic @ points.reshape(-1, 4).T).T
        points = points[:, :3]
        return points

    from fs2r.utils.meshcat import create_visualizer, visualize_pointcloud
    vis = create_visualizer()

    obs_real = env.get_obs()
    H, W = obs_real["depth"].shape
    points = depth_to_points(obs_real["depth"].reshape(H, W), obs_real["camera_intrinsic"].reshape(3, 3), obs_real["camera_extrinsic"].reshape(4, 4), depth_scale=1000.)
    points = points.reshape(H, W, 3)
    colors = obs_real["rgb"].reshape(H, W, 3) / 255.

    visualize_pointcloud(
        vis, 'points_real',
        pc=points,
        color=colors * 255.,
        # color=np.ones_like(points) * [0., 0., 255.],
        size=0.01
    )

    with h5py.File(data_path, "r", swmr=True) as f:
        open_loop_obs = {k: v[:] for k, v in f["data"][f"demo_{demo_idx}"]["obs"].items()}

    obs_sim = open_loop_obs
    B, H, W = obs_sim["depth"].shape
    points_sim = depth_to_points(obs_sim["depth"][0].reshape(H, W), obs_sim["camera_intrinsic"][0].reshape(3, 3), obs_sim["camera_extrinsic"][0].reshape(4, 4), depth_scale=1000.)
    points_sim = points_sim.reshape(H, W, 3)
    colors_sim = obs_sim["rgb"][0].reshape(H, W, 3) / 255.

    visualize_pointcloud(
        vis, 'points_sim',
        pc=points_sim,
        # color=colors_sim * 255.,
        color=np.ones_like(points_sim) * [255., 0., 0.],
        size=0.01
    )

def add_vlm_predictions(obs, instructions, timestep, update_every_timesteps=15, model_name="vila_3b_blocks_path_mask_fast", server_ip=None, obs_path=False, obs_mask=False, vlm_cache=None, vlm_cache_step=0):
    if vlm_cache is None or vlm_cache_step < np.floor(timestep / update_every_timesteps).astype(int):
        vlm_cache_step = np.floor(timestep / update_every_timesteps).astype(int)
        print("Querying VLM at timestep", timestep, "...")
        image, path_pred, mask_pred = vila_inference_api(obs["rgb"], instructions[-1], model_name=model_name, server_ip=server_ip, prompt_type="path_mask")
        
        plt.imsave(f"vlm_image_{timestep}.png", image)

        vlm_cache = {
            "path_pred": path_pred,
            "mask_pred": mask_pred,
        }
    
    if obs_path:
        obs["path_vlm"] = vlm_cache["path_pred"]
        obs["path"] = vlm_cache["path_pred"]
    if obs_mask:
        obs["mask_vlm"] = vlm_cache["mask_pred"]
        obs["mask"] = vlm_cache["mask_pred"]

    return obs, vlm_cache, vlm_cache_step
    
def eval_3dda(
    task,
    data_path,
    real_data_path=None,

    policy=None,
    model_config=None,
    ckpt_path=None,

    mode="closed_loop",
    action_chunking=True,
    action_chunk_size=8,

    n_rollouts=10,
    n_steps=112,

    obs_path=False,
    obs_mask=False,
    obs_mask_w_path=False,
    obs_gt=False,
    open_loop_obs_key="obs",
    model_name_vlm="vila_3b_path_mask_fast",
    server_ip_vlm=None,
    update_every_timesteps_vlm=25,
    real=False,
):
    
    if mode == "open_loop":
        action_chunking = False

    # load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path is not None and (policy is None or model_config is None):
        policy, _, _, _, wandb_dict, model_config = load_checkpoint(ckpt_path)
        policy = policy.to(device)

    clip_embedder = CLIPTextEmbedder()
    clip_embedder = clip_embedder.to(device)

    # load env + normalization data
    with h5py.File(data_path, "r", swmr=True) as f:
        env_config = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
        action_min, action_max = (
            f["data"].attrs["actions_min"],
            f["data"].attrs["actions_max"],
        )

    # set to ood seed if not using path or mask
    seed = env_config["seed"] + 1

    to_be_replaced = "/home/memmelma/Projects/robotic/franka_emika_panda"
    if to_be_replaced in env_config["xml_path"]:
        env_config["xml_path"] = env_config["xml_path"].replace(
            to_be_replaced,
            "/gscratch/weirdlab/memmelma/simvla/pick_data_gen/franka_emika_panda",
        )
    
    # init framestack
    num_frames = model_config.history
    framestack = FrameStackWrapper(num_frames=num_frames)

    successes = []
    videos = []
    instructions = []
    for i in trange(n_rollouts, desc="ROLLOUT"):

        # load open_loop and replay data
        if (
            mode == "open_loop"
            or mode == "replay"
        ):
            demo_idx = i
            with h5py.File(
                real_data_path if real_data_path is not None else data_path,
                "r",
                swmr=True,
            ) as f:
                open_loop_obs = {
                    k: v[:]
                    for k, v in f["data"][f"demo_{demo_idx}"][open_loop_obs_key].items()
                }
                open_loop_actions = f["data"][f"demo_{demo_idx}"]["actions"][:]

            with h5py.File(data_path, "r", swmr=True) as f:
                # set obj poses and colors
                obj_poses = f["data"][f"demo_{demo_idx}"]["obs"]["obj_poses"][:]
                obj_pose = obj_poses[0]
                obj_colors = f["data"][f"demo_{demo_idx}"]["obs"]["obj_colors"][:]
                obj_color = obj_colors[0]

                n_steps = open_loop_actions.shape[0] - 1

        if real:
            from fs2r import BASE_DIR
            from fs2r.robot_env import RobotEnv
            from fs2r.utils.pointcloud import read_calibration_file

            calib_file = os.path.join(BASE_DIR, "perception/calibrations", "most_recent_calib.json")
            print("CALIB FILE", calib_file)
            calib_dict = read_calibration_file(calib_file)
            env_config = {
                "address": "172.16.0.1",
                "port": 5050,
                "camera_index": 0,
                "obs_keys": ["lang_instr", "qpos", "qpos_normalized", "gripper_state_discrete", "rgb", "depth", "camera_intrinsic", "camera_extrinsic"],
                "calib_dict": calib_dict,
                "foundation_stereo": True,
                "img_resize": (128, 128),
            }
            env = RobotEnv(**env_config)
            env.set_lang_instr("pick up the blue cube")
            
            
            
            env.set_lang_instr("put blue cube on the red cube")
            # env.set_lang_instr("put red cube on the green cube")



            if True:
                visualize_pointcloud(env, data_path)
                import IPython; IPython.embed()
        else:
            env = CubeEnv(**env_config)
            env.seed(seed + i)

        obs = env.reset()

        if (
            mode == "open_loop"
            or mode == "replay"
        ):
            env.set_obj_poses(obj_pose)
            env.set_obj_colors(obj_color)

        obs["lang_instr"] = clip_embedder.embed_instruction(obs["lang_instr"])

        instructions.append(env.get_lang_instr())

        # # HACK swap in real observations
        # data_path = "fullrollout.hdf5"
        # demo_idx = 0
        # with h5py.File(data_path, "r", swmr=True) as f:
        #     open_loop_obs = {
        #         k: v[:] for k, v in f["data"][f"demo_{demo_idx}"]["obs_real"].items()
        #     }
        #     n_steps = open_loop_obs["rgb"].shape[0] - 1
        # # HACK swap in real observations
        # Example:
        # python run_3dda_eval.py --name 3dda_low_res_fast_fps_2_h_2 --mode open_loop --n_steps 2 --ckpt best --dataset gifs_curobo/pick_1000_1_objs_128_s2r.hdf5 --n_rollouts 1

        # imgs = [obs["rgb"]]
        imgs = [np.concatenate([obs["rgb"], env.get_obs()["rgb"]], axis=1)]
        if mode == "open_loop":
            obs = {
                k: v[0] for k, v in open_loop_obs.items() if k in env_config["obs_keys"]
            }

        if server_ip_vlm is None and (obs_path or obs_mask or obs_mask_w_path):
            if obs_path:
                obs["path" if obs_gt else "path_vlm"] = open_loop_obs["path" if obs_gt else "path_vlm"][0]
            if obs_mask or obs_mask_w_path:
                obs["mask" if obs_gt else "mask_vlm"] = open_loop_obs["mask" if obs_gt else "mask_vlm"][0]
        elif obs_path or obs_mask or obs_mask_w_path:
            # initial vlm predictions and cache
            obs, vlm_cache, vlm_cache_step = add_vlm_predictions(obs, instructions, timestep=0, update_every_timesteps=update_every_timesteps_vlm, model_name=model_name_vlm, server_ip=server_ip_vlm, obs_path=obs_path, obs_mask=obs_mask or obs_mask_w_path)
        
        framestack.reset()
        framestack.add_obs(obs)

        pred_actions = []
        assert n_steps % action_chunk_size == 0
        for j in range(n_steps // action_chunk_size):

            obs = framestack.get_obs_history()

            act_queue = []
            if mode == "replay":
                act = open_loop_actions[j]
                act_queue.append(act)
            else:
                # add batch dimension, convert to torch
                sample = {
                    "obs": {
                        k: torch.from_numpy(v[None]).to(device) for k, v in obs.items()
                    }
                }
                # preprocess same as training
                batch_prepared = prepare_batch(
                    sample,
                    history=model_config.history,
                    horizon=model_config.horizon,
                    obs_crop=model_config.obs_crop,
                    obs_crop_cube=model_config.obs_crop_cube,
                    obs_noise_std=0.0,
                    obs_path=obs_path,
                    obs_mask=obs_mask,
                    obs_mask_w_path=obs_mask_w_path,
                    obs_outlier=False, # real,
                    obs_gt=obs_gt,
                    device=device,
                )

                if obs_path:
                    rgb_obs = (
                        batch_prepared["rgb_obs"][0, 0].permute(1, 2, 0).cpu().numpy()
                        * 255.0
                    )
                    imgs.append(rgb_obs.astype(np.uint8))
                # if mask_mode is not None:
                #     depth_obs = batch_prepared["pcd_obs"][0,0].permute(1,2,0).cpu().numpy()
                #     imgs.append(depth_obs)

                # [B, T, 7+1]
                with torch.no_grad():
                    acts = policy.forward(**batch_prepared, run_inference=True)

                if action_chunking:
                    chunk_size = action_chunk_size
                else:
                    chunk_size = 1
                for i, act in enumerate(acts[0].cpu().numpy()[:chunk_size]):
                    # discretize gripper action to ensure gripper_state is (0., 1.) as during data gen
                    act[7] = 1.0 if act[7] > 0.5 else 0.0
                    act_queue.append(act)

            for t, act in zip(reversed(range(len(act_queue))), act_queue):
                pred_actions.append(act)
                
                obs_keys_copy = env.obs_keys.copy()
                # only render depth for history in real to speed up inference
                if real and t >= model_config.history:
                    env.obs_keys.remove("rgb")
                    env.obs_keys.remove("depth")
                    env.obs_keys.remove("camera_intrinsic")
                    env.obs_keys.remove("camera_extrinsic")
                    
                obs, r, done, info = env.step(act)
                env.obs_keys = obs_keys_copy

                obs["lang_instr"] = clip_embedder.embed_instruction(obs["lang_instr"])

                if mode == "open_loop":
                    obs = {
                        k: v[j + 1]
                        for k, v in open_loop_obs.items()
                        if k in env_config["obs_keys"]
                    }
                if server_ip_vlm is None and (obs_path or obs_mask or obs_mask_w_path):
                    if obs_path:
                        obs["path" if obs_gt else "path_vlm"] = open_loop_obs["path" if obs_gt else "path_vlm"][0]
                    if obs_mask or obs_mask_w_path:
                        obs["mask" if obs_gt else "mask_vlm"] = open_loop_obs["mask" if obs_gt else "mask_vlm"][0]

                elif obs_path or obs_mask or obs_mask_w_path:
                    # initial vlm predictions and cache
                    timestep=j*action_chunk_size
                    update_every_timesteps=update_every_timesteps_vlm
                    if not "rgb" in obs.keys() and vlm_cache_step < np.floor(timestep / update_every_timesteps).astype(int):
                        obs["rgb"] = env.get_obs()["rgb"]
                    # update vlm predictions and cache -> only compute new path/mask predictions every 15 steps
                    obs, vlm_cache, vlm_cache_step = add_vlm_predictions(obs, instructions, timestep=j*action_chunk_size, update_every_timesteps=update_every_timesteps_vlm, model_name=model_name_vlm, server_ip=server_ip_vlm, obs_path=obs_path, obs_mask=obs_mask or obs_mask_w_path, vlm_cache=vlm_cache, vlm_cache_step=vlm_cache_step)

                framestack.add_obs(obs)
                if not real:
                    imgs.append(env.get_obs()["rgb"])

                success = env.is_success(task=task)
                if success:
                    break
            if success:
                break

        successes.append(success)
        try:
            videos.append(np.array(imgs))
        except:
            videos.append(np.array([im[:128, :128, :3] for im in imgs]))

        if ckpt_path is not None and (mode == "open_loop" or mode == "replay"):
            save_dir = os.path.join(os.path.dirname(ckpt_path), mode)
            os.makedirs(save_dir, exist_ok=True)
            plot_actions(
                np.stack(pred_actions),
                denormalize(open_loop_actions, min=action_min, max=action_max),
                file_name=os.path.join(save_dir, f"img_{i}"),
                act_dim_labels=[
                    "joint0",
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "grasp",
                ],
            )

    print(
        f"{mode} {'act chunking' if action_chunking else ''} Success rate: {np.mean(successes)}"
    )
    return successes, videos, instructions


if __name__ == "__main__":

    # args for ckpt_path and data_path using argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="3dda_closeup")
    parser.add_argument("--task", type=str, default="pick_and_place")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument("--ckpt_dir", type=str, default=None)

    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/memmelma/Projects/robotic/gifs_curobo/red_cube_5000_closeup.hdf5",
    )
    parser.add_argument("--mode", type=str, default="closed_loop")
    parser.add_argument(
        "--no_action_chunking", action="store_true", help="Enable action chunking"
    )
    parser.add_argument(
        "--real", action="store_true", help="Use real robot"
    )
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--obs_path", action="store_true", help="Use path observations")
    parser.add_argument("--obs_mask", action="store_true", help="Use mask observations")
    parser.add_argument("--obs_mask_w_path", action="store_true", help="Use mask observations with path")
    parser.add_argument("--server_ip_vlm", type=str, default=None)
    parser.add_argument("--update_every_timesteps_vlm", type=int, default=25)
    args = parser.parse_args()

    if args.ckpt_dir is None:
        ckpt_path = (
            f"/home/marius/Projects/problem_reduction/results/{args.name}/{args.ckpt}.pth"
        )
    else:
        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt + ".pth")

    successes, videos, instructions = eval_3dda(
        task=args.task,
        data_path=args.dataset,
        ckpt_path=ckpt_path,
        mode=args.mode,
        action_chunking=not args.no_action_chunking,
        action_chunk_size=args.action_chunk_size,
        n_rollouts=args.n_rollouts,
        n_steps=args.n_steps,
        obs_path=args.obs_path,
        obs_mask=args.obs_mask,
        obs_mask_w_path=args.obs_mask_w_path,
        server_ip_vlm=args.server_ip_vlm,
        real=args.real,
        update_every_timesteps_vlm=args.update_every_timesteps_vlm,
    )

    save_dir = os.path.join(os.path.dirname(ckpt_path), args.mode, args.ckpt)
    print(
        "RESULT",
        "ckpt",
        ckpt_path,
        "success rate",
        np.mean(successes),
        "saved to",
        save_dir,
    )
    os.makedirs(save_dir, exist_ok=True)
    for i, (video, instruction) in enumerate(zip(videos, instructions)):
        imageio.mimsave(os.path.join(save_dir, f"img_{instruction}_{i}.gif"), video)
