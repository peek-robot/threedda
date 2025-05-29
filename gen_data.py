import os
import torch
import imageio
from tqdm import trange
import numpy as np

from utils.mp import CuroboWrapper
from utils.robot_env import CubeEnv
from utils.blocks import compute_grasp_pose
from utils.collector import DataCollector


def plan_pick_motion(obj_pose, mp, qpos=None, ee_pose=None):

    segments = []
    obj_pos, obj_quat = obj_pose

    # plan pick motion
    grasp_pos, grasp_quat = compute_grasp_pose(obj_pos, obj_quat)

    target = {
        "ee_pos": torch.from_numpy(grasp_pos).float().cuda()[None]
        + torch.tensor([[0, 0, 0.107]]).cuda(),
        "ee_quat": torch.from_numpy(grasp_quat).float().cuda()[None],
    }

    if ee_pose is not None:
        start = {
            "ee_pos": ee_pose[0],
            "ee_quat": ee_pose[1],
        }
    else:
        start = {"qpos": torch.from_numpy(qpos).float().cuda()[None]}

    traj = mp.plan_motion(start, target)
    segments.append(traj.position.cpu().numpy())

    # plan retract motion

    start = {
        # "ee_pos": target["ee_pos"],
        # "ee_quat": target["ee_quat"],
        "qpos": torch.from_numpy(segments[-1][-1])
        .float()
        .cuda()[None]
    }
    target = {
        "ee_pos": target["ee_pos"] + torch.tensor([[0, 0, 0.2]]).cuda(),
        "ee_quat": target["ee_quat"],
    }
    traj = mp.plan_motion(start, target)
    segments.append(traj.position.cpu().numpy())

    return segments


def plan_motion(obj_pose, mp, qpos=None, ee_pose=None):

    obj_pos, obj_quat = obj_pose

    # plan pick motion
    grasp_pos, grasp_quat = compute_grasp_pose(obj_pos, obj_quat)

    target = {
        "ee_pos": torch.from_numpy(grasp_pos).float().cuda()[None]
        + torch.tensor([[0, 0, 0.107]]).cuda(),
        "ee_quat": torch.from_numpy(grasp_quat).float().cuda()[None],
    }

    if ee_pose is not None:
        start = {
            "ee_pos": ee_pose[0],
            "ee_quat": ee_pose[1],
        }
    else:
        start = {
            "qpos": (
                torch.from_numpy(qpos).float().cuda()[None]
                if qpos is not None
                else None
            ),
        }

    traj = mp.plan_motion(start, target)
    return traj.position.cpu().numpy()


if __name__ == "__main__":

    n_episodes = 2500
    save_dir = "/home/memmelma/Projects/robotic/gifs_curobo"
    outfile = f"debug_pick_red_cube_{n_episodes}.hdf5"

    env_config = {
        "xml_path": "/home/memmelma/Projects/robotic/franka_emika_panda/scene.xml",
        "num_objs": 1,
        "size": 0.03,
        "obj_pos_dist": [[0.4, -0.1, 0.03], [0.6, 0.1, 0.03]],
        "obj_ori_dist": [[0, 0], [0, 0], [-np.pi / 4, np.pi / 4]],
        "obj_color_dist": [[1, 0, 0], [1, 0, 0]],
        "seed": 0,
        "n_steps": 50,
        "time_steps": 0.002,
    }
    env = CubeEnv(**env_config)

    data_config = {
        "n_episodes": n_episodes,
        "qpos_noise_std": 0.0,
        "train_valid_split": 0.99,
    }
    mp = CuroboWrapper(interpolation_dt=env.n_steps * env.time_steps)

    data_collector = DataCollector(
        env,
        env_config,
        data_config,
        save_dir=save_dir,
        out_file=outfile,
        train_valid_split=data_config["train_valid_split"],
        obs_keys=[
            "rgb",
            "depth",
            "camera_intrinsic",
            "camera_extrinsic",
            "qpos",
            "obj_poses",
        ],
    )

    for i in trange(n_episodes):

        env.reset_objs()
        data_collector.reset()

        # get initial state
        obj_poses = env.get_obj_poses()
        obj_pos, obj_quat = obj_poses[0][:3], obj_poses[0][3:7]
        qpos = env.get_qpos()
        pos, quat = mp.compute_fk(torch.from_numpy(qpos).float().cuda())

        # plan pick motion
        segments = plan_pick_motion(qpos=qpos, obj_pose=(obj_pos, obj_quat), mp=mp)

        # execute
        for qpos in segments[0]:
            noise = np.random.normal(0, data_config["qpos_noise_std"], size=qpos.shape)
            act = np.concatenate((qpos + noise, [255.0]))
            data_collector.step(act)
        for qpos in segments[1]:
            noise = np.random.normal(0, data_config["qpos_noise_std"], size=qpos.shape)
            act = np.concatenate((qpos + noise, [0.0]))
            data_collector.step(act)

        # imgs = np.array(data_collector.obs["rgb"])
        # imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)

        data_collector.save()

    data_collector.close()
