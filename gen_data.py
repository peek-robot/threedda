import os
import torch
import imageio
from tqdm import trange
import numpy as np

from utils.mp import CuroboWrapper
from utils.robot_env import CubeEnv
from utils.collector import DataCollector

from utils.mp_helper import plan_pick_motion

def visualize_points(env):
    from utils.meshcat import create_visualizer, visualize_pointcloud
    vis = create_visualizer()

    points = env.get_points()
    colors = env.get_rgb().reshape(-1, 3)
    visualize_pointcloud(
        vis, 'points',
        pc=points,
        color=colors,
        size=0.01
    )
    import IPython; IPython.embed()
    
if __name__ == "__main__":

    n_episodes = 1000
    save_dir = "/home/memmelma/Projects/robotic/gifs_curobo"
    
    outfile = f"blue_cube_{n_episodes}_fast_path_mask.hdf5"

    from utils.pointclouds import read_calibration_file
    # calib_file = "/home/memmelma/Projects/robotic/most_recent_calib_realsense.json"
    calib_file = "/home/memmelma/Projects/robotic/most_recent_calib_zed.json"
    calib_dict = read_calibration_file(calib_file)
    
    env_config = {
        # CubeEnv
        "xml_path": "/home/memmelma/Projects/robotic/franka_emika_panda/scene.xml",
        "num_objs": 1,
        "size": 0.03,
        # # large randomization
        # "obj_pos_dist": [[0.4, -0.1, 0.03], [0.6, 0.1, 0.03]],
        # "obj_ori_dist": [[0, 0], [0, 0], [-np.pi / 4, np.pi / 4]],
        # # medium randomization
        "obj_pos_dist": [[0.4, -0.1, 0.03], [0.6, 0.1, 0.03]],
        "obj_ori_dist": [[0, 0], [0, 0], [0, 0]],
        # # small randomization
        # "obj_pos_dist": [[0.5, -0.1, 0.03], [0.6, 0.1, 0.03]],
        # "obj_ori_dist": [[0, 0], [0, 0], [0, 0]],
        # # no randomization
        # "obj_pos_dist": [[0.5, 0.0, 0.03], [0.5, 0.0, 0.03]],
        # "obj_ori_dist": [[0, 0], [0, 0], [0, 0]],
        # "obj_color_dist": [[0, 0, 1], [0, 0, 1]],
        "seed": 0,
        "obs_keys": ["ee_pos", "ee_pose", "qpos", "qpos_normalized", "gripper_state_discrete", "gripper_state_continuous", "gripper_state_normalized", "obj_poses", "obj_colors", "rgb", "depth", "camera_intrinsic", "camera_extrinsic"],
        
        # RobotEnv
        "camera_name": "custom",
        # render in higher res to ensure full scene is captured
        "img_render": [480, 480],
        # resize to lower res to reduce memory usage and speed up training
        "img_resize": [128, 128],
        "calib_dict": calib_dict,
        "n_steps": 50,
        "time_steps": 0.002,
        "reset_qpos_noise_std": 1e-2,
        "controller": "abs_joint",
    }
    env = CubeEnv(**env_config)

    data_config = {
        "n_episodes": n_episodes,
        "visual_augmentation": False,
        "action_noise_std": 0.0,
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
    )

    if data_config["visual_augmentation"]:
        env.init_randomize()

    for i in trange(n_episodes):
        
        if data_config["visual_augmentation"]:
            env.randomize()
    
        env.reset_objs()
        if data_config["visual_augmentation"] and env.num_objs == 1:
            env.set_obj_colors(np.clip(np.array([0.,0.,0.7]) + np.random.uniform(0.0, 0.3, size=3), 0.0, 1.0))


        data_collector.reset()

        # visualize_points(env)

        # get initial state
        obj_poses = env.get_obj_poses()
        obj_pos, obj_quat = obj_poses[:3], obj_poses[3:7]
        qpos = env.get_qpos()
        prev_qpos = env.get_qpos()

        # plan pick motion
        segments = plan_pick_motion(qpos=qpos, obj_pose=(obj_pos, obj_quat), mp=mp)

        from typing import List
        def subsample_min_velocity(qpos: np.ndarray, min_velocity: float, req_indices: List[int]):
            indices = [0]
            last_idx = 0
            for i in range(1, len(qpos)):
                if i == req_indices[0]:
                    indices.append(i)
                    last_idx = i
                    continue
                velocity = np.linalg.norm(qpos[i] - qpos[last_idx])
                if velocity >= min_velocity:
                    indices.append(i)
                    last_idx = i
            return indices
        
        last_indices = []
        cumsum = 0
        for segment in segments:
            cumsum += len(segment)
            last_indices.append(cumsum)

        grippers = np.concatenate([np.ones(len(segments[0])), np.ones(len(segments[1])), np.zeros(len(segments[2]))])
        qposs = np.concatenate(segments)
        indices = subsample_min_velocity(qposs, 0.05, last_indices)
        
        for qpos, gripper in zip(qposs[indices], grippers[indices]):
            noise = np.random.normal(0, data_config["action_noise_std"], size=qpos.shape)
            if env_config["controller"] == "rel_joint":
                act = np.concatenate((qpos - prev_qpos + noise, [gripper]))
            else:
                act = np.concatenate((qpos + noise, [gripper]))
            data_collector.step(act)
            prev_qpos = env.get_qpos()

        # # execute
        # # skip some steps to speed up execution
        # for qpos in segments[0][::2]:
        #     noise = np.random.normal(0, data_config["action_noise_std"], size=qpos.shape)
        #     if env_config["controller"] == "rel_joint":
        #         act = np.concatenate((qpos - prev_qpos + noise, [1.0]))
        #     else:
        #         act = np.concatenate((qpos + noise, [1.0]))
        #     data_collector.step(act)
        #     prev_qpos = env.get_qpos()

        # # grasp
        # for qpos in segments[1][::2]:
        #     noise = np.random.normal(0, data_config["action_noise_std"], size=qpos.shape)
        #     if env_config["controller"] == "rel_joint":
        #         act = np.concatenate((qpos - prev_qpos + noise, [1.0]))
        #     else:
        #         act = np.concatenate((qpos + noise, [1.0]))
        #     data_collector.step(act)
        #     prev_qpos = env.get_qpos()

        # for qpos in segments[2][::2]:
        #     noise = np.random.normal(0, data_config["action_noise_std"], size=qpos.shape)
        #     if env_config["controller"] == "rel_joint":
        #         act = np.concatenate((qpos - prev_qpos + noise, [0.0]))
        #     else:
        #         act = np.concatenate((qpos + noise, [0.0]))
        #     data_collector.step(act)
        #     prev_qpos = env.get_qpos()

        # import matplotlib.pyplot as plt
        # qpos = np.array(data["qpos"])
        # act = np.array(data["act"])
        # prev_qpos = np.array(data["prev_qpos"])
        # fig, axs = plt.subplots(7, 1, figsize=(10, 10), sharex=True)
        # for j in range(qpos.shape[1]):
        #     axs[j].plot(prev_qpos[:, j], label=f"prev_qpos_{j}")
        #     axs[j].plot(qpos[:, j], label=f"qpos_{j}")
        #     axs[j].plot(act[:, j], label=f"act_{j}")
        # axs[0].legend()
        # plt.savefig(os.path.join(save_dir, f"img_{i}.png"))
        # import IPython; IPython.embed()

        if i % 100 == 0 or i < 10:
            imgs = np.array(data_collector.obs["rgb"])
            imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)

        data_collector.save()

    data_collector.close()
