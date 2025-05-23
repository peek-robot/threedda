import os
import time
import torch
import imageio

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.mp import CuroboWrapper
from utils.robot_env import RobotEnv
from utils.blocks import compute_grasp_pose
from utils.blocks import add_objects_to_mujoco_xml

def plan_pick_motion(ee_pose, obj_pose, mp):
    
    segments = []
    obj_pos, obj_quat = obj_pose

    # plan pick motion
    grasp_pos, grasp_quat = compute_grasp_pose(obj_pos, obj_quat)

    target_pos_torch = torch.from_numpy(grasp_pos).float().cuda()[None] + torch.tensor([[0, 0, 0.107]]).cuda()
    target_quat_torch = torch.from_numpy(grasp_quat).float().cuda()[None]

    pos, quat = ee_pose
    traj = mp.plan_motion(pos, quat, target_pos_torch, target_quat_torch)
    segments.append(traj.position.cpu().numpy())
    
    # plan retract motion
    pos, quat = target_pos_torch, target_quat_torch
    target_pos_torch = target_pos_torch + torch.tensor([[0, 0, 0.2]]).cuda()
    traj = mp.plan_motion(pos, quat, target_pos_torch, target_quat_torch)
    segments.append(traj.position.cpu().numpy())

    return segments

class DataCollector:
    def __init__(self, env, save_dir):
        self.env = env
        self.save_dir = save_dir
        self.obs_keys = ["rgb", "qpos", "obj_pose"]
        self.obs = {key: [] for key in self.obs_keys}
        self.actions = {"qpos": [], "gripper_pos": []}

    def step(self, qpos, gripper_pos):
        self.env.step(qpos, gripper_pos=gripper_pos)
        self.actions["qpos"].append(qpos)
        self.actions["gripper_pos"].append(gripper_pos)
        for key in self.obs_keys:
            self.obs[key].append(getattr(self.env, f"get_{key}")())

    def save(self):
        pass

if __name__ == "__main__":

    n_episodes = 10
    save_dir = "/home/memmelma/Projects/robotic/gifs_curobo"

    num_objs = 1
    mass = 0.05
    size = 0.03

    # add objects
    # model_path = "/home/memmelma/Projects/mjctrl/franka_emika_panda/scene.xml"
    root_dir = "/home/memmelma/Projects/mjctrl/franka_emika_panda"
    obj_names = [f"cube_{i}" for i in range(num_objs)]
    colors = np.random.uniform([0, 0, 0], [1, 1, 1], size=(len(obj_names), 3))
    modified_xml = add_objects_to_mujoco_xml(os.path.join(root_dir, "scene.xml"), num_objs=num_objs, mass=mass, size=size, colors=colors)
    with open(os.path.join(root_dir, "tmp.xml"), "w") as f:
        f.write(modified_xml)

    env = RobotEnv(model_path=os.path.join(root_dir, "tmp.xml"), obj_names=obj_names)
    
    interpolation_dt = env.n_steps * env.time_steps # 0.1
    mp = CuroboWrapper(interpolation_dt=interpolation_dt)

    for i in range(n_episodes):

        start = time.time()

        data_collector = DataCollector(env, save_dir)

        env.reset()

        obj_poses = []
        for _ in range(num_objs):
            box_pos = np.random.uniform([0.4, -0.1, 0.03], [0.6, 0.1, 0.03])
            box_euler = np.zeros(3)
            box_euler[2] = np.random.uniform(np.pi/4, -np.pi/4)
            box_quat = R.from_euler("xyz", box_euler, degrees=False).as_quat(scalar_first=True)
            obj_poses.append(np.concatenate((box_pos, box_quat)))
        env.set_obj_pose(obj_poses)


        obj_poses = env.get_obj_pose()
        obj_pos, obj_quat = obj_poses[0][:3], obj_poses[0][3:7]
        qpos = env.get_qpos()
        pos, quat = mp.compute_fk(torch.from_numpy(qpos).float().cuda())

        segments = plan_pick_motion(ee_pose=(pos, quat), obj_pose=(obj_pos, obj_quat), mp=mp)

        for qpos in segments[0]:
            data_collector.step(qpos, gripper_pos=255.0)

        for qpos in segments[1]:
            data_collector.step(qpos, gripper_pos=0.0)

        end = time.time()
        print(f"Time taken: {end - start}")

        imgs = np.array(data_collector.obs["rgb"])
        imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)