import os
import torch
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R



class RobotEnv:
    def __init__(self, model_path, n_steps=50, time_steps=0.002, obj_names=[]):
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        gravity_compensation = True
        self.model.body_gravcomp[:] = float(gravity_compensation)
        self.time_steps = time_steps
        self.model.opt.timestep = time_steps
        self.n_steps = n_steps

        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.joint_qpos_ids = [self.model.joint(name).id for name in joint_names]
        self.joint_actuator_ids = [self.model.actuator(name).id for name in joint_names]

        self.gripper_qpos_ids = [self.model.joint(name).id for name in ["finger_joint1", "finger_joint2"]]
        self.gripper_actuator_ids = [self.model.actuator(name).id for name in ["joint8"]]

        self.obj_qpos_ids = [self.model.joint(name).id for name in obj_names]
        
        # self.reset_qpos = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.])
        self.reset_qpos = np.array([ 0.01159181, -0.52629131,  0.01545296, -2.77785516, -0.0103323,   3.00063801,  0.7462694])

        self.img_height = 480
        self.img_width = 480
        self.renderer = mujoco.Renderer(self.model, height=self.img_height, width=self.img_width)

    def set_obj_pose(self, obj_poses):
        for obj_qpos_id, obj_pose in zip(self.obj_qpos_ids, obj_poses):
            self.data.qpos[self.model.jnt_qposadr[obj_qpos_id] : self.model.jnt_qposadr[obj_qpos_id] + 7] = np.hstack(obj_pose)
        mujoco.mj_forward(self.model, self.data)

    def get_obj_pose(self):
        obj_poses = []
        for obj_qpos_id in self.obj_qpos_ids:
            obj_poses.append(self.data.qpos[self.model.jnt_qposadr[obj_qpos_id] : self.model.jnt_qposadr[obj_qpos_id] + 7])
        return obj_poses

    def get_qpos(self):
        return self.data.qpos[self.joint_qpos_ids].astype(np.float32)

    def reset(self):
        self.data.qpos[self.joint_qpos_ids] = self.reset_qpos
        self.data.qpos[self.gripper_qpos_ids] = [0.04, 0.04]
        mujoco.mj_forward(self.model, self.data)
        # self.step(self.reset_qpos, gripper_pos=255.0)

    def render(self, camera_name="front", modality="rgb"):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.renderer._scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE
        self.renderer.update_scene(self.data, camera=cam_id)

        if modality == "rgb":
            rgb = self.renderer.render()
            return rgb
        elif modality == "depth":
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()
            return depth
    
    def get_rgb(self):
        return self.render(modality="rgb")
    
    def get_depth(self):
        return self.render(modality="depth")

    def get_points(self):
        from utils.pointclouds import depth_to_points
        depth = self.get_depth()
        intrinsic = self.get_camera_intrinsic("front")
        extrinsic = self.get_camera_extrinsic("front")
        points = depth_to_points(depth, intrinsic=intrinsic, extrinsic=extrinsic, depth_scale=1.0)
        return points

    def step(self, joint_pos, gripper_pos):
        # Apply the action to the robot.
        self.data.ctrl[self.joint_actuator_ids] = joint_pos
        self.data.ctrl[self.gripper_actuator_ids] = gripper_pos
        mujoco.mj_step(self.model, self.data, nstep=self.n_steps)

    def get_camera_intrinsic(self, camera_name):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        fovy = self.model.cam_fovy[cam_id]

        fy = self.img_height / (2 * np.tan(np.radians(fovy / 2)))
        fx = fy
        cx = self.img_width / 2
        cy = self.img_height / 2

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return K


    def get_camera_extrinsic(self, camera_name):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        camera_pos = self.data.cam_xpos[cam_id]
        camera_rot = self.data.cam_xmat[cam_id].reshape(3, 3)

        R = np.eye(4)
        R[:3, :3] = camera_rot
        R[:3, 3] = camera_pos

        # https://github.com/ARISE-Initiative/robosuite/blob/de64fa5935f9f30ce01b36a3ef1a3242060b9cdb/robosuite/utils/camera_utils.py#L39
        camera_axis_correction = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        R = R @ camera_axis_correction

        return R