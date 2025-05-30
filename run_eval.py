import json
import h5py
import torch
import imageio
import os
import numpy as np
from tqdm import trange

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from utils.framestack import FrameStackWrapper
from utils.robot_env import CubeEnv
from utils.collector import denorm_actions

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

    mode = "closed_loop" # "closed_loop", "open_loop", "replay"

    n_rollouts = 10
    n_steps = 70

    # # ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/rgb_trans_hist/20250530102349/models/model_epoch_250.pth"
    # # data_path = "/home/memmelma/Projects/robotic/gifs_curobo/pick_red_cube_2500_all.hdf5"
    # ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/state_trans_rand/20250530104422/models/model_epoch_250.pth"
    # data_path = "/home/memmelma/Projects/robotic/gifs_curobo/pick_red_cube_2500_random_all.hdf5"

    # # ???
    # # ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/pcd_trans_hist/20250530104422/models/model_epoch_250.pth"
    # data_path ="/home/memmelma/Projects/robotic/gifs_curobo/pick_red_cube_100_all_closeup.hdf5"

    ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/rgb_trans_rand/20250530113319/models/model_epoch_350.pth"
    data_path = "/home/memmelma/Projects/robotic/gifs_curobo/pick_red_cube_2500_random_all.hdf5"

    ckpt_path = "/home/memmelma/Projects/robotic/robomimic_pcd/robomimic/../bc_transformer_trained_models/pcd_trans_hist/20250530115542/models/model_epoch_130.pth"
    data_path = "/home/memmelma/Projects/robotic/gifs_curobo/pick_red_cube_100_all_closeup.hdf5"

    save_dir = "."

    # load policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    
    # load data
    demo_idx = 0
    with h5py.File(data_path, "r", swmr=True) as f:
        env_config = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
        action_min, action_max = f["data"].attrs["actions_min"], f["data"].attrs["actions_max"]
        
        if mode == "open_loop" or mode == "replay":
            open_loop_obs = {k: v[:] for k, v in f["data"][f"demo_{demo_idx}"]["obs"].items()}
            open_loop_actions = f["data"][f"demo_{demo_idx}"]["raw_actions"][:]
            n_steps = open_loop_actions.shape[0] - 1

    # init env
    env_config["seed"] += 1
    env = CubeEnv(**env_config)
    
    # init framestack
    num_frames = json.loads(ckpt_dict["config"])["train"]["frame_stack"]
    framestack = FrameStackWrapper(num_frames=num_frames)

    for i in trange(n_rollouts):

        # reset everything
        env.reset_objs()
        obs = env.reset()
        
        if mode == "open_loop":
            obs = {k: v[0] for k, v in open_loop_obs.items()}
        
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

            act_denorm = denorm_actions(act, action_min, action_max)
            pred_actions.append(act_denorm)
            
            obs, r, done, info = env.step(act_denorm)

            if mode == "open_loop":
                obs = {k: v[j+1] for k, v in open_loop_obs.items()}
            obs = ObsUtils.process_obs_dict(obs)
            framestack.add_obs(obs)
        
        if mode == "open_loop":
            plot_actions(np.stack(pred_actions), open_loop_actions, file_name=f"img_{i}", act_dim_labels=["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "grasp"])
        imgs = np.array(imgs)
        imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)
