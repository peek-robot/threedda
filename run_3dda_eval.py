import json
import h5py
import torch
import imageio
import os
import numpy as np
from tqdm import trange

from threedda.text_embed import CLIPTextEmbedder
from threedda.utils import load_checkpoint, prepare_batch

from utils.framestack import FrameStackWrapper
from utils.robot_env import CubeEnv
from utils.normalize import denormalize

import matplotlib.pyplot as plt
def plot_actions(pred_actions, true_actions, file_name, act_dim_labels=["x", "y", "z", "yaw", "pitch", "roll", "grasp"]):
    """
    Plots predicted vs. ground truth actions (7-dim) along with a corresponding image strip.
    Logs the plot to WandB.
    """
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplot_mosaic([act_dim_labels])
    fig.set_size_inches([40, 5])

    # Ensure proper input formatting for actions
    pred_actions = np.array(pred_actions).squeeze()  # Bx7
    true_actions = np.array(true_actions).squeeze()  # Bx7

    # Plot actions for each dimension
    for action_dim, action_label in enumerate(act_dim_labels):
        axs[action_label].plot(pred_actions[:, action_dim], label="Predicted")
        axs[action_label].plot(true_actions[:, action_dim], label="Ground Truth")
        axs[action_label].set_title(action_label)
        axs[action_label].set_xlabel("Time (steps)")
        axs[action_label].legend()

    plt.tight_layout()
    # wandb.log({wandb_title: wandb.Image(fig)})
    plt.savefig(f"{file_name}.png")
    # plt.close(fig)

if __name__ == "__main__":

    mode = "closed_loop" # closed_loop" # "closed_loop", "open_loop", "replay"

    n_rollouts = 10
    n_steps = 50 # 70

    action_chunking = True
    if mode == "open_loop":
        action_chunking = False

    # works w/ action chunking
    ckpt_path = "/home/memmelma/Projects/robotic/results/3dda_closeup/best.pth"
    # 
    
    ckpt_path = "/home/memmelma/Projects/robotic/results/3dda_new_params/best.pth"
    data_path = "/home/memmelma/Projects/robotic/gifs_curobo/red_cube_5000_closeup.hdf5"
    
    ckpt_path = "/home/memmelma/Projects/robotic/results/3dda_no_noise/last.pth"
    ckpt_path = "/home/memmelma/Projects/robotic/results/3dda_no_noise/best.pth"
    data_path = "/home/memmelma/Projects/robotic/gifs_curobo/blue_cube_1000_path_no_noise.hdf5"
    

    save_dir = os.path.join(os.path.dirname(ckpt_path), mode)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_embedder = CLIPTextEmbedder()
    clip_embedder = clip_embedder.to(device)

    # load policy
    policy, _, _, _, wandb_dict, model_config = load_checkpoint(ckpt_path)
    policy = policy.to(device)

    # load data
    with h5py.File(data_path, "r", swmr=True) as f:
        env_config = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
        action_min, action_max = f["data"].attrs["actions_min"], f["data"].attrs["actions_max"]

    # init env
    env_config["seed"] += 1
    print("WARNING: evaluating in train env")
    env_config["obs_keys"] = ["qpos", "gripper_state", "rgb", "depth", "camera_intrinsic", "camera_extrinsic"]
    env = CubeEnv(**env_config)
    
    # init framestack
    num_frames = model_config.history
    framestack = FrameStackWrapper(num_frames=num_frames)

    successes = []
    for i in trange(n_rollouts):

        
        # load data
        if mode == "open_loop" or mode == "replay":
            demo_idx = i
            with h5py.File(data_path, "r", swmr=True) as f:
                open_loop_obs = {k: v[:] for k, v in f["data"][f"demo_{demo_idx}"]["obs"].items()}
                open_loop_actions = f["data"][f"demo_{demo_idx}"]["actions"][:]

                obj_poses = f["data"][f"demo_{demo_idx}"]["obs"]["obj_poses"][:]
                obj_pose = obj_poses[0]
                obj_colors = f["data"][f"demo_{demo_idx}"]["obs"]["obj_colors"][:]
                obj_color = obj_colors[0]

                n_steps = open_loop_actions.shape[0] - 1


        # reset everything
        env.reset_objs()
        if mode == "open_loop" or mode == "replay":
            env.set_obj_poses(obj_pose)
            env.set_obj_colors(obj_color)
        obs = env.reset()
        
        if mode == "open_loop":
            obs = {k: v[0] for k, v in open_loop_obs.items() if k in env_config["obs_keys"]}
        
        framestack.add_obs(obs)
        
        imgs = []
        pred_actions = []
        for j in range(n_steps):
            
            # logging
            imgs.append(env.get_rgb())

            obs = framestack.get_obs_history()

            if mode == "replay":
                act = open_loop_actions[j]
            else:
                # add batch dimension, convert to torch
                sample = {"obs": {k: torch.from_numpy(v[None]).to(device) for k, v in obs.items()}}
                # preprocess same as training
                batch_prepared = prepare_batch(sample, clip_embedder, history=model_config.history, horizon=model_config.horizon, obs_noise_std=0.0, device=device)
                # [B, T, 7+1]
                with torch.no_grad():
                    acts = policy.forward(**batch_prepared, run_inference=True)
                
                if action_chunking:
                    for act in acts[0].cpu().numpy():
                        pred_actions.append(act)
                        act[7] = 1. if act[7] > 0.5 else 0.
                        obs, r, done, info = env.step(act)
                else:
                    act = acts.cpu().numpy()[0,0]
                    # discretize gripper action to ensure gripper_state is (0., 1.) as during data gen
                    act[7] = 1. if act[7] > 0.5 else 0.
                    pred_actions.append(act)
                    obs, r, done, info = env.step(act)

            # # act = denormalize(act, min=action_min, max=action_max)
            # pred_actions.append(act)
            
            # obs, r, done, info = env.step(act)

            if mode == "open_loop":
                obs = {k: v[j+1] for k, v in open_loop_obs.items() if k in env_config["obs_keys"]}
            framestack.add_obs(obs)
            if env.is_success(task="pick"):
                break
        successes.append(env.is_success(task="pick"))
        if mode == "open_loop" or mode == "replay":
            plot_actions(np.stack(pred_actions), denormalize(open_loop_actions, min=action_min, max=action_max), file_name=os.path.join(save_dir, f"img_{i}"), act_dim_labels=["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "grasp"])
        imgs = np.array(imgs)
        imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)

    print(f"{mode} {'act chunking' if action_chunking else ''} Success rate: {np.mean(successes)}")