import json
import h5py
import torch
import imageio
import os
import glob
import numpy as np
from tqdm import trange

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

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

    mode = "open_loop" # "closed_loop", "open_loop", "replay"

    n_rollouts = 10
    n_steps = 70

    # RELATIVE JOINT
    data_path = "/home/memmelma/Projects/robotic/gifs_curobo/blue_cube_1000_128_rel_joint.hdf5"
    # 80%
    # ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/gmm_rgb_rel_joint_1k/20250612164954/models/"
    # 100%
    ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/gmm_proprio_rgb_rel_joint_1k/20250612174423/models/"
    # 70%
    # ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/gmm_pcd_1k/20250612171024/models/"
    
    # # ABSOLUTE JOINT
    # data_path = "/home/memmelma/Projects/robotic/gifs_curobo/blue_cube_1000_128_abs_joint.hdf5"
    # # 80
    # ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/gmm_rgb_abs_joint_1k/20250612181557/models/"
    # # 100
    # ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/gmm_proprio_rgb_abs_joint_1k/20250612211526/models/"
    # # 20% -> plays same action
    # ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/gmm_pcd_abs_joint_1k/20250612224946/models/"
    
    # # VIS AUG
    ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/gmm_proprio_rgb_rel_vis_aug/20250613115711/models"
    # ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/gmm_rgb_rel_vis_aug/20250613115549/models/"

    # ckpt = "model_epoch_500.pth"
    ckpt = "model_epoch_400.pth"
    ckpt_path = os.path.join(ckpt_path, ckpt)
    save_dir = "."
    
    # load policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    
    # load data
    with h5py.File(data_path, "r", swmr=True) as f:
        env_config = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
        action_min, action_max = f["data"].attrs["actions_min"], f["data"].attrs["actions_max"]

    # init env
    env_config["seed"] += 1
    print("WARNING: evaluating in train env")
    env_config["obs_keys"] = ckpt_dict["shape_metadata"]["all_obs_keys"]

    # env_config["controller"] = "abs_joint"
    env = CubeEnv(**env_config)
    
    # init framestack
    num_frames = json.loads(ckpt_dict["config"])["train"]["frame_stack"]
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
        
        obs = ObsUtils.process_obs_dict(obs)
        framestack.add_obs(obs)
        
        policy.start_episode()

        imgs = []
        pred_actions = []
        for j in range(n_steps):
            
            # logging
            imgs.append(env.get_rgb())

            obs = framestack.get_obs_history()

            if mode == "replay":
                act = open_loop_actions[j]
            else:
                act = policy(ob=obs)
            act = denormalize(act, min=action_min, max=action_max)
            pred_actions.append(act)
            
            obs, r, done, info = env.step(act)

            if mode == "open_loop":
                obs = {k: v[j+1] for k, v in open_loop_obs.items() if k in env_config["obs_keys"]}
            obs = ObsUtils.process_obs_dict(obs)
            framestack.add_obs(obs)
            if env.is_success(task="pick"):
                break
        successes.append(env.is_success(task="pick"))
        if mode == "open_loop" or mode == "replay":
            plot_actions(np.stack(pred_actions), denormalize(open_loop_actions, min=action_min, max=action_max), file_name=os.path.join(save_dir, f"img_{i}.png"), act_dim_labels=["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "grasp"])
        imgs = np.array(imgs)
        imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)

    print(f"Success rate: {np.mean(successes)}")