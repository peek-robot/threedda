import os
import torch
import imageio
from tqdm import tqdm
import numpy as np

from utils.mp import CuroboWrapper
from utils.robot_env import CubeEnv
from utils.collector import DataCollector

from utils.mp_helper import plan_pick_and_place_motion

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

    n_episodes = 10000
    save_dir = "/home/memmelma/Projects/robotic/gifs_curobo"
    # outfile = f"blue_cube_{n_episodes}_128.hdf5"
    outfile = f"pnp_{n_episodes}_128_abs_joint.hdf5"

    from utils.pointclouds import read_calibration_file
    calib_file = "/home/memmelma/Projects/robotic/most_recent_calib.json"
    calib_dict = read_calibration_file(calib_file)
    
    env_config = {
        # CubeEnv
        "xml_path": "/home/memmelma/Projects/robotic/franka_emika_panda/scene.xml",
        "num_objs": 2,
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
        "obj_color_dist": [[0.5, 0.5, 0.5], [1, 1, 1]],
        "seed": 0,
        "obs_keys": ["ee_pos", "ee_pose","qpos", "qpos_normalized", "gripper_state", "obj_poses", "obj_colors", "rgb", "depth", "camera_intrinsic", "camera_extrinsic"],
        
        # RobotEnv
        "camera_name": "custom",
        # render in higher res to ensure full scene is captured
        "img_render": [480, 480],
        # resize to lower res to reduce memory usage and speed up training
        "img_resize": [128, 128], # [224, 224],
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
        "action_noise_std": 0.0, # 5e-3,
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

    pbar = tqdm(range(n_episodes))
    i = 0
    while i < n_episodes:
        
        if data_config["visual_augmentation"]:
            env.randomize()
        env.reset_objs()

        data_collector.reset()

        # visualize_points(env)

        # get initial state
        obj_poses = env.get_obj_poses()
        pick_pos, pick_quat = obj_poses[:3], obj_poses[3:7]
        place_pos, place_quat = obj_poses[7:10], obj_poses[10:13]
        place_pos[2] += env_config["size"] * 2

        qpos = env.get_qpos()
        prev_qpos = env.get_qpos()

        # plan pick motion
        segments = plan_pick_and_place_motion(qpos=qpos, obj_pose=(pick_pos, pick_quat), place_pose=(place_pos, place_quat), mp=mp)

        # execute
        # skip some steps to speed up execution
        for qpos in segments[0]:
            noise = np.random.normal(0, data_config["action_noise_std"], size=qpos.shape)
            if env_config["controller"] == "rel_joint":
                act = np.concatenate((qpos - prev_qpos + noise, [1.0]))
            else:
                act = np.concatenate((qpos + noise, [1.0]))
            data_collector.step(act)
            prev_qpos = env.get_qpos()
        for qpos in segments[1]:
            noise = np.random.normal(0, data_config["action_noise_std"], size=qpos.shape)
            if env_config["controller"] == "rel_joint":
                act = np.concatenate((qpos - prev_qpos + noise, [0.0]))
            else:
                act = np.concatenate((qpos + noise, [0.0]))
            data_collector.step(act)
            prev_qpos = env.get_qpos()
        for qpos in segments[2]:
            noise = np.random.normal(0, data_config["action_noise_std"], size=qpos.shape)
            if env_config["controller"] == "rel_joint":
                act = np.concatenate((qpos - prev_qpos + noise, [0.0]))
            else:
                act = np.concatenate((qpos + noise, [0.0]))
            data_collector.step(act)
            prev_qpos = env.get_qpos()

        if not env.is_success("pick_and_place"):
            data_collector.reset()
            continue
        i += 1
        pbar.update(1)
        
        if i <= 10:
            imgs = np.array(data_collector.obs["rgb"])
            imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)

        data_collector.save()

    data_collector.close()
