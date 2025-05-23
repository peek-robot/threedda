import os
import time
import torch
import imageio

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.mp import CuroboWrapper
from utils.robot_env import RobotEnv
from utils.blocks import compute_grasp_pose, compute_all_grasp_poses
from utils.blocks import add_objects_to_mujoco_xml

if __name__ == "__main__":

    n_episodes = 50
    save_dir = "/home/memmelma/Projects/robotic/gifs_curobo"

    interpolation_dt = 0.1
    mp = CuroboWrapper(interpolation_dt=interpolation_dt)


    num_objs = 1
    mass = 0.05
    size = 0.03

    for i in range(n_episodes):

        start = time.time()

        # add objects
        # model_path = "/home/memmelma/Projects/mjctrl/franka_emika_panda/scene.xml"
        root_dir = "/home/memmelma/Projects/mjctrl/franka_emika_panda"
        obj_names = [f"cube_{i}" for i in range(num_objs)]
        colors = np.random.uniform([0, 0, 0], [1, 1, 1], size=(len(obj_names), 3))
        modified_xml = add_objects_to_mujoco_xml(os.path.join(root_dir, "scene.xml"), num_objs=num_objs, mass=mass, size=size, colors=colors)
        with open(os.path.join(root_dir, "tmp.xml"), "w") as f:
            f.write(modified_xml)

        env = RobotEnv(model_path=os.path.join(root_dir, "tmp.xml"), obj_names=obj_names, n_steps=int(interpolation_dt/0.002))

        imgs = []

        env.reset()

        # from utils.pointclouds import crop_points
        # from utils.meshcat import create_visualizer, visualize_pointcloud

        # rgb = env.get_rgb()
        # points = env.get_points()
        # points, colors = crop_points(points, colors=rgb.reshape(-1, 3), crop_min=[0., -0.3, 0.], crop_max=[0.6, 0.3, 1.0])
        # vis = create_visualizer()
        # visualize_pointcloud(
        #     vis, 'points',
        #     pc=points,
        #     color=colors,
        #     size=0.01
        # )
    
        imgs.append(env.get_rgb())


        obj_poses = env.get_obj_pose()
        qpos = env.get_qpos()
        pos, quat = mp.compute_fk(torch.from_numpy(qpos).float().cuda())

        grasp_pos, grasp_quat = compute_grasp_pose(obj_poses[0][:3], obj_poses[0][3:7])
        grasp_pos = torch.from_numpy(grasp_pos).float().cuda()[None] + torch.tensor([[0, 0, 0.107]]).cuda()
        grasp_quat = torch.from_numpy(grasp_quat).float().cuda()[None]
        traj = mp.plan_motion(pos, quat, grasp_pos, grasp_quat)


        
        # # compute grasp poses #1
        # obj_poses = env.get_obj_pose()
        # qpos = env.get_qpos()
        # pos, quat = mp.compute_fk(torch.from_numpy(qpos).float().cuda())

        # grasp_pos, grasp_quat = compute_all_grasp_poses(obj_poses[0][:3], obj_poses[0][3:7])
        # dists = []
        # for g_quat in grasp_quat:
        #     g_R = R.from_quat(g_quat, scalar_first=True)
        #     curr_R = R.from_quat(quat.cpu().numpy()[0], scalar_first=True)
        #     dist = g_R.inv() * curr_R
        #     dists.append(dist.as_quat(scalar_first=True))

        # def smallest_quaternion_diff(quat_diffs):
        #     angles = [2 * np.arccos(np.clip(q[0], -1.0, 1.0)) for q in quat_diffs]
        #     return np.argmin(angles)

        # idx = smallest_quaternion_diff(dists)
        # grasp_pos, grasp_quat = grasp_pos[idx], grasp_quat[idx]

        # # plan motion #1
        # grasp_pos = torch.from_numpy(grasp_pos).float().cuda()[None] + torch.tensor([[0, 0, 0.107]]).cuda()
        # grasp_quat = torch.from_numpy(grasp_quat).float().cuda()[None]
        # traj = mp.plan_motion(pos, quat, grasp_pos, grasp_quat)

        # grasp_pos, grasp_quat = compute_all_grasp_poses(obj_poses[0][:3], obj_poses[0][3:7])
        # grasp_pos = np.stack(grasp_pos)
        # grasp_quat = np.stack(grasp_quat)
        # grasp_pos = torch.from_numpy(grasp_pos).float().cuda()[None] + torch.tensor([[0, 0, 0.107]]).cuda()
        # grasp_quat = torch.from_numpy(grasp_quat).float().cuda()[None]
        # traj = mp.plan_motion_set(pos, quat, grasp_pos, grasp_quat)

        # execute #1
        for qpos in traj.position.cpu().numpy():
            env.step(qpos, gripper_pos=255.0)
            imgs.append(env.get_rgb())
        

        # execute GRASP
        env.step(qpos, gripper_pos=0.0)


        # compute retract pose #2
        pos, quat = mp.compute_fk(torch.from_numpy(qpos).float().cuda())
        grasp_pos = grasp_pos + torch.tensor([[0, 0, 0.2]]).cuda()

        # plan motion #2
        traj = mp.plan_motion(pos, quat, grasp_pos, grasp_quat)

        # execute #2
        for qpos in traj.position.cpu().numpy():
            env.step(qpos, gripper_pos=0.0)
            imgs.append(env.get_rgb())

        end = time.time()
        print(f"Time taken: {end - start}")

        # imageio.mimsave(os.path.join(save_dir, f"img_{i}.gif"), imgs)
