import os
import torch
import imageio
from tqdm import trange
import numpy as np

from utils.mp import CuroboWrapper
from utils.robot_env import CubeEnv
from utils.collector import DataCollector

from utils.mp_helper import plan_pick_motion

if __name__ == "__main__":

    n_episodes = 5000
    save_dir = "/home/memmelma/Projects/robotic/gifs_curobo"
    # outfile = f"pick_red_cube_{n_episodes}_all_random_closeup.hdf5"
    outfile = f"pick_red_cube_{n_episodes}_low_random_real_cam.hdf5"

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
        "obj_color_dist": [[1, 0, 0], [1, 0, 0]],
        "seed": 0,
        "obs_keys": ["qpos", "obj_poses", "obj_colors", "rgb", "depth", "camera_intrinsic", "camera_extrinsic"],
        
        # RobotEnv
        "camera_name": "custom",
        "img_height": 480,
        "img_width": 640,
        "calib_dict": calib_dict,
        "n_steps": 50,
        "time_steps": 0.002,
    }
    env = CubeEnv(**env_config)

    data_config = {
        "n_episodes": n_episodes,
        "qpos_noise_std": 0.003,
        "train_valid_split": 0.9,
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

        # get initial state
        obj_poses = env.get_obj_poses()
        obj_pos, obj_quat = obj_poses[:3], obj_poses[3:7]
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

        if n_episodes <= 10:
            imgs = np.array(data_collector.obs["rgb"])
            imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)

        data_collector.save()

    data_collector.close()
