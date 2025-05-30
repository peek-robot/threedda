import os

os.environ["MUJOCO_GL"] = "egl"
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.blocks import add_objects_to_mujoco_xml


class RobotEnv:
    def __init__(self, model_path, camera_name="front", img_height=256, img_width=256, calib_dict=None, n_steps=50, time_steps=0.002):

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        gravity_compensation = True
        self.model.body_gravcomp[:] = float(gravity_compensation)
        self.time_steps = time_steps
        self.model.opt.timestep = time_steps
        self.n_steps = n_steps

        joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.joint_qpos_ids = [self.model.joint(name).id for name in joint_names]
        self.joint_actuator_ids = [self.model.actuator(name).id for name in joint_names]

        self.gripper_qpos_ids = [
            self.model.joint(name).id for name in ["finger_joint1", "finger_joint2"]
        ]
        self.gripper_actuator_ids = [
            self.model.actuator(name).id for name in ["joint8"]
        ]

        # self.reset_qpos = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.])
        self.reset_qpos = np.array(
            [
                0.01159181,
                -0.52629131,
                0.01545296,
                -2.77785516,
                -0.0103323,
                3.00063801,
                0.7462694,
            ]
        )

        self.camera_name = camera_name
        self.img_height = img_height
        self.img_width = img_width
        self.calib_dict = calib_dict
        
        if self.calib_dict is not None:

            for sn, camera_name in zip(self.calib_dict.keys(), [self.camera_name]):
                self.set_camera_intrinsic(
                    camera_name,
                    self.calib_dict[sn]["intrinsic"]["fx"],
                    self.calib_dict[sn]["intrinsic"]["fy"],
                    self.calib_dict[sn]["intrinsic"]["ppx"],
                    self.calib_dict[sn]["intrinsic"]["ppy"],
                    self.calib_dict[sn]["intrinsic"]["fovy"],
                )

            for sn, camera_name in zip(self.calib_dict.keys(), [self.camera_name]):
                R = np.eye(4)
                R[:3, :3] = np.array(self.calib_dict[sn]["extrinsic"]["ori"])
                R[:3, 3] = np.array(self.calib_dict[sn]["extrinsic"]["pos"]).reshape(-1)
                self.set_camera_extrinsic(camera_name, R)

        self.renderer = mujoco.Renderer(
            self.model, height=self.img_height, width=self.img_width
        )
                
        self.action_dimension = 7 + 1
        
    def get_qpos(self):
        return self.data.qpos[self.joint_qpos_ids].astype(np.float32)

    def reset(self):
        # let objects settle w/o physics
        mujoco.mj_step(self.model, self.data, nstep=self.n_steps)
        # set robot state
        self.data.qpos[self.joint_qpos_ids] = self.reset_qpos
        self.data.qpos[self.gripper_qpos_ids] = [0.04, 0.04]
        # apply robot state w/o physics
        mujoco.mj_forward(self.model, self.data)

    def render(self, modality="rgb"):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
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
        return (self.render(modality="depth") * 1000).astype(np.int16)
        # return self.render(modality="depth")

    def get_points(self):
        from utils.pointclouds import depth_to_points

        depth = self.get_depth()
        intrinsic = self.get_camera_intrinsic(self.camera_name)
        extrinsic = self.get_camera_extrinsic(self.camera_name)
        points = depth_to_points(
            depth, intrinsic=intrinsic, extrinsic=extrinsic, depth_scale=1000.0
        )
        # points = depth_to_points(depth, intrinsic=intrinsic, extrinsic=extrinsic, depth_scale=1.0)
        return points

    def step(self, action):
        # Apply the action to the robot.
        self.data.ctrl[self.joint_actuator_ids] = action[:7]
        self.data.ctrl[self.gripper_actuator_ids] = action[7]
        mujoco.mj_step(self.model, self.data, nstep=self.n_steps)

    def set_camera_intrinsic(self, camera_name, fx, fy, cx, cy, fovy):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.model.cam_fovy[cam_id] = np.degrees(
            2 * np.arctan(self.img_height / (2 * fy))
        )

    def set_camera_extrinsic(self, camera_name, R):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        cam_base_pos = R[:3, 3]
        cam_base_ori = R[:3, :3]
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )

        self.model.cam_pos[cam_id] = cam_base_pos
        from scipy.spatial.transform import Rotation as Rot
        self.model.cam_quat[cam_id] = Rot.from_matrix(cam_base_ori @ camera_axis_correction).as_quat(scalar_first=True)

    def get_camera_intrinsic(self):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        fovy = self.model.cam_fovy[cam_id]

        fy = self.img_height / (2 * np.tan(np.radians(fovy / 2)))
        fx = fy
        cx = self.img_width / 2
        cy = self.img_height / 2

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return K

    def get_camera_extrinsic(self):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)

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


class CubeEnv(RobotEnv):
    def __init__(
        self,
        xml_path,
        num_objs=1,
        size=0.03,
        obj_pos_dist=[[0.4, -0.1, 0.03], [0.6, 0.1, 0.03]],
        obj_ori_dist=[[0, 0], [0, 0], [-np.pi / 4, np.pi / 4]],
        obj_color_dist=[[0, 0, 0], [1, 1, 1]],
        obs_keys=["qpos", "rgb"],
        seed=0,
        **kwargs,
    ):
        self.seed = seed
        self.obs_keys = obs_keys

        self.num_objs = num_objs
        self.size = size
        self.obj_pos_dist = obj_pos_dist
        self.obj_ori_dist = obj_ori_dist
        self.obj_color_dist = obj_color_dist
        np.random.seed(self.seed)
        modified_xml_path = self.generate_xml(xml_path, self.num_objs, self.size)
        super().__init__(modified_xml_path, **kwargs)

        self.obj_names = [f"cube_{i}" for i in range(num_objs)]
        self.obj_qpos_ids = [self.model.joint(name).id for name in self.obj_names]
        self.obj_geom_ids = [self.model.geom(name).id for name in self.obj_names]

    def step(self, action):
        super().step(action)
        return self.get_obs(), 0, False, {}
    
    def reset(self):
        super().reset()
        return self.get_obs()

    def get_obs(self):
        obs = {}
        for key in self.obs_keys:
            obs[key] = getattr(self, f"get_{key}")()
        return obs
    
    def reset_objs(self):
        obj_poses = []
        for _ in range(self.num_objs):
            box_pos = np.random.uniform(self.obj_pos_dist[0], self.obj_pos_dist[1])
            box_euler = np.zeros(3)
            box_euler[2] = np.random.uniform(
                self.obj_ori_dist[2][0], self.obj_ori_dist[2][1]
            )
            box_quat = R.from_euler("xyz", box_euler, degrees=False).as_quat(
                scalar_first=True
            )
            obj_poses.append(np.concatenate((box_pos, box_quat)))
        self.set_obj_poses(obj_poses)
        colors = np.random.uniform(
            self.obj_color_dist[0], self.obj_color_dist[1], size=(self.num_objs, 3)
        )
        self.set_obj_colors(colors)

    def set_obj_poses(self, obj_poses):
        for obj_qpos_id, obj_pose in zip(self.obj_qpos_ids, obj_poses):
            self.data.qpos[
                self.model.jnt_qposadr[obj_qpos_id] : self.model.jnt_qposadr[
                    obj_qpos_id
                ]
                + 7
            ] = np.hstack(obj_pose)
        mujoco.mj_forward(self.model, self.data)

    def set_obj_colors(self, obj_colors):
        for obj_geom_id, obj_color in zip(self.obj_geom_ids, obj_colors):
            self.model.geom_rgba[obj_geom_id] = np.concatenate((obj_color, [1.0]))
        mujoco.mj_forward(self.model, self.data)

    def get_obj_poses(self):
        obj_poses = []
        for obj_qpos_id in self.obj_qpos_ids:
            obj_poses.append(
                self.data.qpos[
                    self.model.jnt_qposadr[obj_qpos_id] : self.model.jnt_qposadr[
                        obj_qpos_id
                    ]
                    + 7
                ]
            )
        return np.concatenate(obj_poses, axis=0)

    def generate_xml(self, xml_path, num_objs, size):
        colors = np.random.uniform([0, 0, 0], [1, 1, 1], size=(num_objs, 3))
        modified_xml = add_objects_to_mujoco_xml(
            xml_path,
            num_objs=num_objs,
            mass=0.05,
            size=size,
            colors=colors,
            orientation=False,
        )
        modified_xml_path = os.path.join(os.path.dirname(xml_path), "tmp.xml")
        with open(modified_xml_path, "w") as f:
            f.write(modified_xml)
        return modified_xml_path

    def is_success(self, task):
        if task == "pick":
            return self.get_obj_poses()[0][2] > 0.1
        else:
            raise ValueError(f"Invalid task: {task}")