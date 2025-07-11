import json
import h5py
import torch
import imageio
import os
import numpy as np
from tqdm import trange

from problem_reduction.threedda.text_embed import CLIPTextEmbedder
from problem_reduction.threedda.data import prepare_batch
from problem_reduction.threedda.model import load_checkpoint

from problem_reduction.robot.wrappers.framestack import FrameStackWrapper
from problem_reduction.robot.robot_env import CubeEnv
from problem_reduction.utils.normalize import denormalize

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
    real_data_path=None,
    policy=None,
    model_config=None,
    clip_embedder=None,
    ckpt_path=None,
    mode="closed_loop",
    action_chunking=True,
    action_chunk_size=8,
    n_rollouts=None,
    n_steps=None,
    path_mode=None,  # "open_loop",
    mask_mode=None,  # "open_loop",
    open_loop_obs_key="obs",
):

    if "pick_and_place" in data_path:
        task = "pick_and_place"
    elif "pick" in data_path:
        task = "pick"
    else:
        raise ValueError(f"Invalid task: {task}")

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

    # load env + normalization data
    with h5py.File(data_path, "r", swmr=True) as f:
        env_config = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
        action_min, action_max = (
            f["data"].attrs["actions_min"],
            f["data"].attrs["actions_max"],
        )

    # set to ood seed if not using path or mask
    if path_mode is None and mask_mode is None:
        env_config["seed"] += 1

    # env_config["obs_keys"] = ["qpos", "gripper_state_continuous", "gripper_state_discrete", "rgb", "depth", "camera_intrinsic", "camera_extrinsic"]
    to_be_replaced = "/home/memmelma/Projects/robotic/franka_emika_panda"
    if to_be_replaced in env_config["xml_path"]:
        env_config["xml_path"] = env_config["xml_path"].replace(
            to_be_replaced,
            "/gscratch/weirdlab/memmelma/simvla/pick_data_gen/franka_emika_panda",
        )
    env = CubeEnv(**env_config)

    # init framestack
    num_frames = model_config.history
    framestack = FrameStackWrapper(num_frames=num_frames)

    if (n_rollouts is None or n_steps is None) and task == "pick_and_place":
        n_steps = 90
        n_rollouts = 10
    elif (n_rollouts is None or n_steps is None) and task == "pick":
        n_steps = 70
        n_rollouts = 25

    successes = []
    videos = []
    instructions = []
    for i in trange(n_rollouts, desc="ROLLOUT"):

        # load open_loop and replay data
        if (
            mode == "open_loop"
            or mode == "replay"
            or path_mode is not None
            or mask_mode is not None
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

        # reset everything
        # env.reset_objs()

        obs = env.reset()
        if (
            mode == "open_loop"
            or mode == "replay"
            or path_mode is not None
            or mask_mode is not None
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
        if path_mode is not None:
            obs["path_vlm"] = open_loop_obs["path_vlm"][0]
        if mask_mode is not None:
            obs["mask_vlm"] = open_loop_obs["mask_vlm"][0]
        framestack.add_obs(obs)

        pred_actions = []
        for j in range(n_steps):

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
                    clip_embedder,
                    history=model_config.history,
                    horizon=model_config.horizon,
                    obs_crop=model_config.obs_crop,
                    obs_crop_cube=model_config.obs_crop_cube,
                    obs_noise_std=0.0,
                    obs_path=path_mode is not None and path_mode == "open_loop",
                    obs_mask=mask_mode is not None and mask_mode == "open_loop",
                    device=device,
                )

                if path_mode is not None:
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
                    for i, act in enumerate(acts[0].cpu().numpy()[:action_chunk_size]):
                        # discretize gripper action to ensure gripper_state is (0., 1.) as during data gen
                        act[7] = 1.0 if act[7] > 0.5 else 0.0
                        act_queue.append(act)
                else:
                    act = acts.cpu().numpy()[0, 0]
                    # discretize gripper action to ensure gripper_state is (0., 1.) as during data gen
                    act[7] = 1.0 if act[7] > 0.5 else 0.0
                    act_queue.append(act)

            for t, act in zip(reversed(range(len(act_queue))), act_queue):
                pred_actions.append(act)
                # # HACK: only render when required
                # if t < num_frames:
                #     obs_keys_copy = env.obs_keys
                #     env.obs_keys = ["rgb"]

                obs, r, done, info = env.step(act)
                obs["lang_instr"] = clip_embedder.embed_instruction(obs["lang_instr"])

                if mode == "open_loop":
                    obs = {
                        k: v[j + 1]
                        for k, v in open_loop_obs.items()
                        if k in env_config["obs_keys"]
                    }
                if path_mode is not None:
                    obs["path_vlm"] = open_loop_obs["path_vlm"][0]
                if mask_mode is not None:
                    obs["mask_vlm"] = open_loop_obs["mask_vlm"][0]

                # # HACK: only render when required
                # if t < num_frames:
                #     env.obs_keys = obs_keys_copy
                framestack.add_obs(obs)

                # imgs.append(obs["rgb"])
                imgs.append(np.concatenate([obs["rgb"], env.get_obs()["rgb"]], axis=1))

                # check for success
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
    parser.add_argument("--action_chunk_size", type=int, default=8)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=70)
    parser.add_argument("--path_mode", type=str, default=None)
    parser.add_argument("--mask_mode", type=str, default=None)

    args = parser.parse_args()

    if args.ckpt_dir is None:
        ckpt_path = (
            f"/home/memmelma/Projects/robotic/results/{args.name}/{args.ckpt}.pth"
        )
    else:
        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt + ".pth")

    successes, videos, instructions = eval_3dda(
        data_path=args.dataset,
        ckpt_path=ckpt_path,
        mode=args.mode,
        action_chunking=not args.no_action_chunking,
        action_chunk_size=args.action_chunk_size,
        n_rollouts=args.n_rollouts,
        n_steps=args.n_steps,
        path_mode=args.path_mode,
        mask_mode=args.mask_mode,
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
