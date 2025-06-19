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


def plot_actions(
    pred_actions,
    true_actions,
    file_name,
    act_dim_labels=["x", "y", "z", "yaw", "pitch", "roll", "grasp"],
):
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


def eval_3dda(
    data_path,
    policy=None,
    model_config=None,
    clip_embedder=None,
    ckpt_path=None,
    mode="closed_loop",
    action_chunking=False,
    n_rollouts=10,
    n_steps=50,
):

    if mode == "open_loop":
        action_chunking = False

    # load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path is not None and (policy is None or model_config is None):
        policy, _, _, _, wandb_dict, model_config = load_checkpoint(ckpt_path)
        policy = policy.to(device)
    if clip_embedder is None:
        clip_embedder = CLIPTextEmbedder()
        clip_embedder = clip_embedder.to(device)

    # load normalization data
    if mode == "open_loop" or mode == "replay":
        with h5py.File(data_path, "r", swmr=True) as f:
            env_config = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
            action_min, action_max = (
                f["data"].attrs["actions_min"],
                f["data"].attrs["actions_max"],
            )

    # init env
    env_config["seed"] += 1
    # env_config["obs_keys"] = ["qpos", "gripper_state_continuous", "gripper_state_discrete", "rgb", "depth", "camera_intrinsic", "camera_extrinsic"]
    env = CubeEnv(**env_config)

    # init framestack
    num_frames = model_config.history
    framestack = FrameStackWrapper(num_frames=num_frames)

    # HACK
    # data_path = "/home/memmelma/Projects/robotic/blue_cube_black_curtain.hdf5"

    successes = []
    videos = []
    for i in trange(n_rollouts, desc="ROLLOUT"):

        # load open_loop and replay data
        if mode == "open_loop" or mode == "replay":
            demo_idx = i
            with h5py.File(data_path, "r", swmr=True) as f:
                open_loop_obs = {
                    k: v[:] for k, v in f["data"][f"demo_{demo_idx}"]["obs"].items()
                }
                open_loop_actions = f["data"][f"demo_{demo_idx}"]["actions"][:]

                # set obj poses and colors
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

        imgs = [obs["rgb"]]
        if mode == "open_loop":
            obs = {
                k: v[0] for k, v in open_loop_obs.items() if k in env_config["obs_keys"]
            }
        framestack.add_obs(obs)

        pred_actions = []
        for j in range(n_steps):

            obs = framestack.get_obs_history()

            if mode == "replay":
                act = open_loop_actions[j]
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
                    clip_embedder,
                    history=model_config.history,
                    horizon=model_config.horizon,
                    obs_crop=model_config.obs_crop,
                    obs_noise_std=0.0,
                    obs_path=model_config.obs_path,
                    device=device,
                )
                # [B, T, 7+1]
                with torch.no_grad():
                    acts = policy.forward(**batch_prepared, run_inference=True)

                if action_chunking:
                    # HACK: speed up inference by not querying obs when executing action chunks
                    obs_keys_copy = env.obs_keys
                    env.obs_keys = []

                    # execute half the action chunk
                    for act in acts[0].cpu().numpy()[:8]:
                        pred_actions.append(act)
                        # discretize gripper action to ensure gripper_state is (0., 1.) as during data gen
                        act[7] = 1.0 if act[7] > 0.5 else 0.0
                        obs, r, done, info = env.step(act)

                    env.obs_keys = obs_keys_copy
                    obs = env.get_obs()
                else:
                    act = acts.cpu().numpy()[0, 0]
                    # discretize gripper action to ensure gripper_state is (0., 1.) as during data gen
                    act[7] = 1.0 if act[7] > 0.5 else 0.0
                    pred_actions.append(act)
                    obs, r, done, info = env.step(act)

            imgs.append(obs["rgb"])
            if mode == "open_loop":
                obs = {
                    k: v[j + 1]
                    for k, v in open_loop_obs.items()
                    if k in env_config["obs_keys"]
                }
            framestack.add_obs(obs)

            if env.is_success(task="pick"):
                break

        successes.append(env.is_success(task="pick"))
        videos.append(np.array(imgs))

        if mode == "open_loop" or mode == "replay":
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
    return successes, videos


if __name__ == "__main__":

    # args for ckpt_path and data_path using argparse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="3dda_closeup")
    parser.add_argument("--ckpt", type=str, default="best")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/memmelma/Projects/robotic/gifs_curobo/red_cube_5000_closeup.hdf5",
    )
    parser.add_argument("--mode", type=str, default="closed_loop")
    parser.add_argument(
        "--action_chunking", action="store_true", help="Enable action chunking"
    )
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=70)

    args = parser.parse_args()

    ckpt_path = f"/home/memmelma/Projects/robotic/results/{args.name}/{args.ckpt}.pth"

    successes, videos = eval_3dda(
        data_path=args.dataset,
        ckpt_path=ckpt_path,
        mode=args.mode,
        action_chunking=args.action_chunking,
        n_rollouts=args.n_rollouts,
        n_steps=args.n_steps,
    )

    save_dir = os.path.join(os.path.dirname(ckpt_path), args.mode)
    os.makedirs(save_dir, exist_ok=True)
    for i, video in enumerate(videos):
        imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), video)
