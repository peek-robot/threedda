import os
import h5py
import time
import torch
import imageio
from tqdm import trange
import numpy as np

from scipy.spatial.transform import Rotation as R

from utils.mp import CuroboWrapper
from utils.robot_env import RobotEnv
from utils.blocks import compute_grasp_pose
from utils.blocks import add_objects_to_mujoco_xml
from utils.collector import DataCollector
def plan_pick_motion(obj_pose, mp, qpos=None, ee_pose=None):
    
    segments = []
    obj_pos, obj_quat = obj_pose

    # plan pick motion
    grasp_pos, grasp_quat = compute_grasp_pose(obj_pos, obj_quat)

    target = {
        "ee_pos": torch.from_numpy(grasp_pos).float().cuda()[None] + torch.tensor([[0, 0, 0.107]]).cuda(),
        "ee_quat": torch.from_numpy(grasp_quat).float().cuda()[None]
    }

    if ee_pose is not None:
        start = {
            "ee_pos": ee_pose[0],
            "ee_quat": ee_pose[1],
        }
    else:
        start = {
            "qpos": torch.from_numpy(qpos).float().cuda()[None]
        }

    traj = mp.plan_motion(start, target)
    segments.append(traj.position.cpu().numpy())
    
    # plan retract motion
    
    start = {
        # "ee_pos": target["ee_pos"],
        # "ee_quat": target["ee_quat"],
        "qpos": torch.from_numpy(segments[-1][-1]).float().cuda()[None]
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
        "ee_pos": torch.from_numpy(grasp_pos).float().cuda()[None] + torch.tensor([[0, 0, 0.107]]).cuda(),
        "ee_quat": torch.from_numpy(grasp_quat).float().cuda()[None]
    }

    if ee_pose is not None:
        start = {
            "ee_pos": ee_pose[0],
            "ee_quat": ee_pose[1],
        }
    else:
        start = {
            "qpos": torch.from_numpy(qpos).float().cuda()[None] if qpos is not None else None,
        }

    traj = mp.plan_motion(start, target)
    return traj.position.cpu().numpy()
    
    

if __name__ == "__main__":

    n_episodes = 36
    save_dir = "/home/memmelma/Projects/robotic/gifs_curobo"

    num_objs = 1
    size = 0.03

    qpos_noise_std = 0.# 0.01

    # add objects to mujoco model
    root_dir = "/home/memmelma/Projects/mjctrl/franka_emika_panda"
    obj_names = [f"cube_{i}" for i in range(num_objs)]
    colors = np.random.uniform([0, 0, 0], [1, 1, 1], size=(len(obj_names), 3))
    modified_xml = add_objects_to_mujoco_xml(os.path.join(root_dir, "scene.xml"), num_objs=num_objs, mass=0.05, size=size, colors=colors)
    with open(os.path.join(root_dir, "tmp.xml"), "w") as f:
        f.write(modified_xml)

    # init
    env = RobotEnv(model_path=os.path.join(root_dir, "tmp.xml"), obj_names=obj_names)
    mp = CuroboWrapper(interpolation_dt=env.n_steps * env.time_steps)
    data_collector = DataCollector(env, save_dir, obs_keys=["rgb", "qpos", "obj_poses"], actions_keys=["joint_pos", "gripper_pos"])

    for i in trange(n_episodes):

        # reset objects
        obj_poses = []
        for _ in range(num_objs):
            box_pos = np.random.uniform([0.4, -0.1, size], [0.6, 0.1, size])
            box_euler = np.zeros(3)
            box_euler[2] = np.random.uniform(np.pi/4, -np.pi/4)
            box_quat = R.from_euler("xyz", box_euler, degrees=False).as_quat(scalar_first=True)
            obj_poses.append(np.concatenate((box_pos, box_quat)))
        env.set_obj_poses(obj_poses)
        colors = np.random.uniform([0, 0, 0], [1, 1, 1], size=(len(obj_names), 3))
        env.set_obj_colors(colors)

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
            noise = np.random.normal(0, qpos_noise_std, size=qpos.shape)
            data_collector.step({"joint_pos": qpos + noise, "gripper_pos": 255.0})
        for qpos in segments[1]:
            noise = np.random.normal(0, qpos_noise_std, size=qpos.shape)
            data_collector.step({"joint_pos": qpos + noise, "gripper_pos": 0.0})

        imgs = np.array(data_collector.obs["rgb"])
        imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)

        data_collector.save()
