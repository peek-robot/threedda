import os
import torch
import imageio
from tqdm import trange
import numpy as np

from problem_reduction import ROOT_DIR

from problem_reduction.datagen.mp import CuroboWrapper
from problem_reduction.datagen.mp_helper import (
    plan_pick_motion,
    plan_pick_and_place_motion,
    subsample_min_velocity,
)
from problem_reduction.datagen.collector import DataCollector

from problem_reduction.robot.robot_env import CubeEnv

from problem_reduction.points.meshcat import create_visualizer, visualize_pointcloud
from problem_reduction.points.pointclouds import read_calibration_file

from problem_reduction.threedda.text_embed import CLIPTextEmbedder


def visualize_points(env):
    vis = create_visualizer()

    points = env.get_points()
    colors = env.get_rgb().reshape(-1, 3)
    visualize_pointcloud(vis, "points", pc=points, color=colors, size=0.01)
    import IPython

    IPython.embed()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pick")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_objs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--visual_augmentation", action="store_true")
    parser.add_argument("--drop_failures", action="store_true")
    parser.add_argument("--identifier", type=str, default=None)
    args = parser.parse_args()

    assert args.task == "pick_and_place" and args.num_objs > 1 or args.task == "pick"

    save_dir = "data"

    outfile = f"{args.task}_{args.num_samples}_{str(args.num_objs) + '_objs'}{'_' + 'va' if args.visual_augmentation else ''}{'_' + args.identifier if args.identifier else ''}.hdf5"

    calib_file = os.path.join(
        ROOT_DIR, "robot/calibrations", "real_07_29_09_34.json"
    )
    calib_dict = read_calibration_file(calib_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_embedder = CLIPTextEmbedder(device=device)

    env_config = {
        # CubeEnv
        "xml_path": "robot/sim/franka_emika_panda/scene_new.xml",
        "num_objs": args.num_objs,
        "size": 0.025,
        # "obj_pos_dist": [[0.3, -0.2, 0.03], [0.6, 0.2, 0.03]],
        "obj_pos_dist": [[0.3, -0.2, 0.03], [0.7, 0.2, 0.03]],
        "obj_ori_dist": [[0, 0], [0, 0], [-np.pi / 16, np.pi / 16]],
        "seed": 0,
        "obs_keys": [
            "lang_instr",
            "ee_pos",
            "ee_pose",
            "qpos",
            "qpos_normalized",
            "gripper_state_discrete",
            "gripper_state_continuous",
            "gripper_state_normalized",
            "obj_poses",
            "obj_colors",
            "rgb",
            "depth",
            "camera_intrinsic",
            "camera_extrinsic",
        ],
        # RobotEnv
        "camera_name": "custom",
        # render in higher res to ensure full scene is captured
        "img_render": [480, 480],
        # resize to lower res to reduce memory usage and speed up training
        "img_resize": [128, 128],
        "calib_dict": calib_dict,
        "n_steps": 50,
        "time_steps": 0.002,
        "reset_qpos_noise_std": 2e-2,
        "controller": "abs_joint",
    }
    env = CubeEnv(**env_config)

    data_config = {
        "n_episodes": args.num_samples,
        "visual_augmentation": args.visual_augmentation,
        "action_noise_std": 2e-3, # 2e-3, # 0.0, # 5e-3
        "min_velocity": 0.06,
        "train_valid_split": 0.99 if args.num_samples > 100 else 0.9,
    }
    mp = CuroboWrapper(interpolation_dt=env.n_steps * env.time_steps)

    # hardcode reset qpos using ee space
    reset_qpos = mp.compute_ik(torch.tensor([[0.4, 0.0, 0.3]], device='cuda:0'), torch.tensor([[ 0., 1., 0., 0.]], device='cuda:0'))
    reset_qpos = reset_qpos.cpu().numpy()[0]
    env.reset_qpos = reset_qpos


    data_collector = DataCollector(
        env,
        env_config,
        data_config,
        save_dir=save_dir,
        out_file=outfile,
        train_valid_split=data_config["train_valid_split"],
        lang_key="lang_instr",
        clip_embedder=clip_embedder,
    )

    if args.visual_augmentation:
        env.init_randomize()

    successes = []
    for i in trange(args.num_samples):

        if args.visual_augmentation:
            env.randomize()

        # env.reset_objs()
        # if args.visual_augmentation and env.num_objs == 1:
        #     env.set_obj_colors(np.clip(np.array([0.,0.,0.7]) + np.random.uniform(0.0, 0.3, size=3), 0.0, 1.0))
        data_collector.reset()

        # get initial state
        obj_poses = env.get_obj_poses()
        obj_pos, obj_quat = obj_poses[:3], obj_poses[3:7]
        qpos = env.get_qpos()
        prev_qpos = env.get_qpos()

        # plan motion
        try:
            if args.task == "pick_and_place":
                place_pos, place_quat = obj_poses[7:10], obj_poses[10:14]
                qpos_traj, gripper_traj = plan_pick_and_place_motion(
                    qpos=qpos,
                    obj_pose=(obj_pos, obj_quat),
                    place_pose=(place_pos, place_quat),
                    mp=mp,
                    cube_size=env_config["size"] * 2,
                )
            elif args.task == "pick":
                qpos_traj, gripper_traj = plan_pick_motion(
                    qpos=qpos, obj_pose=(obj_pos, obj_quat), mp=mp
                )
        except ValueError as e:
            print(f"Failed to plan motion: {e}")
            continue

        # make sure end of each segment is preserved
        req_indices = []
        cumsum = 0
        for segment in qpos_traj:
            cumsum += len(segment)
            req_indices.append(cumsum)
        # subsample motion to have minimum velocity
        qpos_traj = np.concatenate(qpos_traj)
        gripper_traj = np.concatenate(gripper_traj)
        indices = subsample_min_velocity(qpos_traj, data_config["min_velocity"], req_indices=req_indices)

        # execute motion
        for qpos, gripper in zip(qpos_traj[indices], gripper_traj[indices]):
            noise = np.random.normal(
                0, data_config["action_noise_std"], size=qpos.shape
            )
            if env_config["controller"] == "rel_joint":
                act = np.concatenate((qpos - prev_qpos + noise, [gripper]))
            else:
                act = np.concatenate((qpos + noise, [gripper]))
            data_collector.step(act)
            prev_qpos = env.get_qpos()

        success = env.is_success(args.task)
        successes.append(success)
        if args.drop_failures and not success:
            continue

        # visualize motion
        if i % 100 == 0 or i < 10:
            imgs = np.array(data_collector.obs["rgb"])
            save_dir_gifs = os.path.join(save_dir, "gifs")
            os.makedirs(save_dir_gifs, exist_ok=True)
            imageio.mimsave(os.path.join(save_dir_gifs, f"img_{i}.gif"), imgs)

        data_collector.save()

    data_collector.close()
    print(f"DONE! Success rate: {np.mean(successes)}")
