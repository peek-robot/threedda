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

    n_episodes = 10 # 250
    save_dir = "/home/memmelma/Projects/robotic/gifs_curobo"
    outfile = f"blue_cube_{n_episodes}_path_no_action_noise.hdf5"

    from utils.pointclouds import read_calibration_file
    calib_file = "/home/memmelma/Projects/robotic/most_recent_calib.json"
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
        "obj_color_dist": [[0, 0, 1], [0, 0, 1]],
        "seed": 0,
        "obs_keys": ["ee_pos", "qpos", "qpos_normalized", "gripper_state", "obj_poses", "obj_colors", "rgb", "depth", "camera_intrinsic", "camera_extrinsic"],
        
        # RobotEnv
        "camera_name": "custom",
        # render in higher res to ensure full scene is captured
        "img_render": [480, 480],
        # resize to lower res to reduce memory usage and speed up training
        "img_resize": [224, 224],
        "calib_dict": calib_dict,
        "n_steps": 50,
        "time_steps": 0.002,
        "reset_qpos_noise_std": 1e-2,
    }
    env = CubeEnv(**env_config)

    data_config = {
        "n_episodes": n_episodes,
        "action_noise_std": 0.0, # 1e-3,
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

    for i in trange(n_episodes):

        env.reset_objs()
        data_collector.reset()

        # visualize_points(env)

        # get initial state
        obj_poses = env.get_obj_poses()
        obj_pos, obj_quat = obj_poses[:3], obj_poses[3:7]
        qpos = env.get_qpos()

        # plan pick motion
        segments = plan_pick_motion(qpos=qpos, obj_pose=(obj_pos, obj_quat), mp=mp)

        # execute
        # skip some steps to speed up execution
        for qpos in segments[0][3:-3]:
            noise = np.random.normal(0, data_config["action_noise_std"], size=qpos.shape)
            act = np.concatenate((qpos + noise, [1.0]))
            data_collector.step(act)
        for qpos in segments[1][3:-3]:
            noise = np.random.normal(0, data_config["action_noise_std"], size=qpos.shape)
            act = np.concatenate((qpos + noise, [0.0]))
            data_collector.step(act)

        if n_episodes <= 10:
            imgs = np.array(data_collector.obs["rgb"])
            imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)

        data_collector.save()

    data_collector.close()
