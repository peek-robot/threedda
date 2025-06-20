import os
import cv2

os.environ["MUJOCO_GL"] = "egl"
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.blocks import add_objects_to_mujoco_xml
from utils.normalize import normalize, denormalize

class RobotEnv:
    def __init__(self, model_path, controller="abs_joint", camera_name="front", img_render=[480, 640], img_resize=None, calib_dict=None, reset_qpos_noise_std=0., n_steps=50, time_steps=0.002):

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.model.body_gravcomp[:] = 1.0
        
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

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        # self.reset_qpos = np.array([0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.])
        
        # isaac pose
        # self.reset_qpos = np.array(
        #     [
        #         0.01159181,
        #         -0.52629131,
        #         0.01545296,
        #         -2.77785516,
        #         -0.0103323,
        #         3.00063801,
        #         0.7462694,
        #     ]
        # )
        
        # much closer
        self.reset_qpos = np.array([
            0.02299943, -0.07843312, -0.03196311, -2.21364984, -0.01667695,
        2.14565732,  0.75160931
        ])

        self.gripper_state = 1.0
        self.reset_qpos_noise_std = reset_qpos_noise_std

        # https://frankaemika.github.io/docs/control_parameters.html
        self.min_qpos = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            dtype=np.float32,
        )
        self.max_qpos = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            dtype=np.float32,
        )

        self.camera_name = camera_name
        self.img_render = img_render
        self.img_resize = img_resize
        self.calib_dict = calib_dict
        self.reset_camera_pose()
        self.renderer = mujoco.Renderer(
            self.model, height=self.img_render[0], width=self.img_render[1]
        )
                
        self.action_dimension = 7 + 1
        self.controller = controller

    def reset_camera_pose(self):
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
        mujoco.mj_forward(self.model, self.data)

    def get_gripper_state_discrete(self):
        return np.array([self.gripper_state], dtype=np.float32)

    def get_gripper_state_continuous(self):
        return np.array([self.data.qpos[self.gripper_qpos_ids][0]], dtype=np.float32)
    
    def get_gripper_state_normalized(self):
        return normalize(self.get_gripper_state_continuous(), min=0.0, max=0.04)
    
    def get_ee_pose(self):
        return np.concatenate((self.get_ee_pos(), self.get_ee_quat()), axis=0)
    
    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].astype(np.float32)

    def get_ee_quat(self):
        try:
            return R.from_matrix(self.get_ee_mat()).as_quat(scalar_first=True)
        except:
            # old scipy version defaults to scalar_first=False
            quat = R.from_matrix(self.get_ee_mat()).as_quat()
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            return quat
    
    def get_ee_mat(self):
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).astype(np.float32)
    
    def get_qpos(self):
        return self.data.qpos[self.joint_qpos_ids].astype(np.float32)

    def get_qpos_normalized(self):
        qpos = self.get_qpos()
        return normalize(qpos, min=self.min_qpos, max=self.max_qpos)

    def reset(self):
        # let objects settle w/o physics
        mujoco.mj_step(self.model, self.data, nstep=self.n_steps)
        # set robot state
        reset_qpos = self.reset_qpos + np.random.normal(0, self.reset_qpos_noise_std, size=self.reset_qpos.shape)
        self.data.qpos[self.joint_qpos_ids] = reset_qpos
        self.data.qpos[self.gripper_qpos_ids] = [0.04, 0.04]
        self.gripper_state = 1.0
        # set velocities to zero after mj_step
        self.data.qvel[self.joint_qpos_ids] = np.zeros(len(self.joint_qpos_ids))
        self.data.qvel[self.gripper_qpos_ids] = np.zeros(len(self.gripper_qpos_ids))
       # apply robot state w/o physics
        mujoco.mj_forward(self.model, self.data)

    def render(self, modality="rgb"):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        # # visualize site frames
        # self.renderer._scene_option.frame = mujoco.mjtFrame.mjFRAME_SITE
        self.renderer.update_scene(self.data, camera=cam_id)

        if modality == "rgb":
            rgb = self.renderer.render()
            return rgb
        elif modality == "depth":
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
            self.renderer.disable_depth_rendering()
            return depth

    def resize_image(self, img):
        img = cv2.resize(img, self.img_resize, interpolation=cv2.INTER_NEAREST)
        return img
    
    def get_rgb(self):
        rgb = self.render(modality="rgb")
        if self.img_resize is not None:
            rgb = self.resize_image(rgb)
        return rgb

    def get_depth(self):
        depth = self.render(modality="depth")
        if self.img_resize is not None:
            depth = self.resize_image(depth)
        return (depth * 1000).astype(np.int16)

    def get_points(self):
        from utils.pointclouds import depth_to_points

        depth = self.get_depth()
        intrinsic = self.get_camera_intrinsic()
        extrinsic = self.get_camera_extrinsic()
        points = depth_to_points(
            depth, intrinsic=intrinsic, extrinsic=extrinsic, depth_scale=1000.0
        )
        # points = depth_to_points(depth, intrinsic=intrinsic, extrinsic=extrinsic, depth_scale=1.0)
        return points

    def compute_ik(self, target_pos, target_quat, integration_dt=0.1, damping=1e-4):
    
        curr_pos, curr_quat = self.get_ee_pos(), self.get_ee_quat()

        # Pre-allocate numpy arrays.
        jac = np.zeros((6, self.model.nv))
        diag = damping * np.eye(6)
        error = np.zeros(6)
        error_pos = error[:3]
        error_ori = error[3:]
        curr_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        # Position error.
        error_pos[:] = target_pos - curr_pos

        # Orientation error.
        mujoco.mju_negQuat(curr_quat_conj, curr_quat)
        mujoco.mju_mulQuat(error_quat, target_quat, curr_quat_conj)
        mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

        # Get the Jacobian with respect to the end-effector site.
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.ee_site_id)
        # Solve system of equations: J @ dq = error.
        jac = jac[:, self.joint_qpos_ids]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

        # Create full joint velocity vector
        dq_full = np.zeros(self.model.nv)
        dq_full[self.joint_qpos_ids] = dq

        # Get full joint position vector
        q_full = self.data.qpos.copy()
        
        # Integrate joint velocities to obtain joint positions
        mujoco.mj_integratePos(self.model, q_full, dq_full, integration_dt)

        # Return only the controlled joint positions
        q = q_full[self.joint_qpos_ids]
        q = np.clip(q, self.min_qpos, self.max_qpos)
        return q

    def step(self, action):
        if self.controller == "abs_joint":
            qpos = action[:7]
        elif self.controller == "rel_joint":
            qpos = self.data.qpos[self.joint_qpos_ids] + action[:7]
        elif self.controller == "abs_ee":
            qpos = self.compute_ik(action[:3], action[3:7])
        elif self.controller == "rel_ee":
            curr_ee_pose = self.get_ee_pose()
            qpos = self.compute_ik(curr_ee_pose[:3] + action[:3], curr_ee_pose[3:] + action[3:7])
        else:
            raise ValueError(f"Invalid controller: {self.controller}")
        gripper_act = action[7]

        # Apply the action to the robot.
        self.data.ctrl[self.joint_actuator_ids] = qpos
        self.data.ctrl[self.gripper_actuator_ids] = gripper_act * 255.0
        self.gripper_state = gripper_act # 1. if action[7] > 0.5 else 0.
        mujoco.mj_step(self.model, self.data, nstep=self.n_steps)
    
    def adjust_intrinsics_for_resize(self, K):
        assert self.img_render[0] == self.img_render[1] and self.img_resize[0] == self.img_resize[1]
        K_new = K.copy()
        scale_x = self.img_resize[1] / float(self.img_render[1])  # width scale
        scale_y = self.img_resize[0] / float(self.img_render[0])  # height scale
        K_new[0, 0] *= scale_x  # fx
        K_new[1, 1] *= scale_y  # fy
        K_new[0, 2] *= scale_x  # cx
        K_new[1, 2] *= scale_y  # cy
        return K_new

    def set_camera_intrinsic(self, camera_name, fx, fy, cx, cy, fovy):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.model.cam_fovy[cam_id] = np.degrees(
            2 * np.arctan(self.img_render[0] / (2 * fy))
        )

    def set_camera_extrinsic(self, camera_name, R, mujoco_format=False):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        cam_base_pos = R[:3, 3]
        cam_base_ori = R[:3, :3]
        if mujoco_format:
            camera_axis_correction = np.eye(3)
        else:
            camera_axis_correction = np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            )

        self.model.cam_pos[cam_id] = cam_base_pos
        from scipy.spatial.transform import Rotation as Rot
        try:
            self.model.cam_quat[cam_id] = Rot.from_matrix(cam_base_ori @ camera_axis_correction).as_quat(scalar_first=True)
        except:
            # old scipy version defaults to scalar_first=False
            quat = Rot.from_matrix(cam_base_ori @ camera_axis_correction).as_quat()
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            self.model.cam_quat[cam_id] = quat

    def get_camera_intrinsic(self):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        fovy = self.model.cam_fovy[cam_id]

        fy = self.img_render[0] / (2 * np.tan(np.radians(fovy / 2)))
        fx = fy
        cx = self.img_render[1] / 2
        cy = self.img_render[0] / 2

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        if self.img_resize is not None:
            K = self.adjust_intrinsics_for_resize(K)
        return K

    def get_camera_extrinsic(self, mujoco_format=False):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)

        camera_pos = self.data.cam_xpos[cam_id]
        camera_rot = self.data.cam_xmat[cam_id].reshape(3, 3)

        R = np.eye(4)
        R[:3, :3] = camera_rot
        R[:3, 3] = camera_pos

        if not mujoco_format:
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

    ### RANDOMIZATION ###
    def init_randomize(self):
        self.reset()
        # camera
        self.calib_dict_copy = self.calib_dict.copy()
        # color
        self.geom_rgba = self.model.geom_rgba.copy()
        # light
        self.light_pos = self.model.light_pos.copy()
        self.light_dir = self.model.light_dir.copy()
        self.light_castshadow = self.model.light_castshadow.copy()
        self.light_ambient = self.model.light_ambient.copy()
        self.light_diffuse = self.model.light_diffuse.copy()
        self.light_specular = self.model.light_specular.copy()

    def reset_randomize(self):
        # color
        self.model.geom_rgba = self.geom_rgba
        # camera pose
        self.calib_dict = self.calib_dict_copy
        self.reset_camera_pose()
        # light
        self.model.light_pos = self.light_pos
        self.model.light_dir = self.light_dir
        self.model.light_castshadow = self.light_castshadow
        self.model.light_ambient = self.light_ambient
        self.model.light_diffuse = self.light_diffuse
        self.model.light_specular = self.light_specular

    def randomize_camera_pose(self):
        # # noise fovy
        # intrinsic = self.get_camera_intrinsic()
        # high_low = 5.
        # fy_noise = np.random.uniform(low=-high_low, high=high_low)
        # intrinsic[1, 1] += fy_noise
        # self.set_camera_intrinsic(self.camera_name, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2], intrinsic[2, 2])
        # noise camera pose
        camera_extrinsic = self.get_camera_extrinsic(mujoco_format=True)
        scale = 2e-2
        pos_noise = np.random.normal(loc=0.0, scale=scale, size=(3,))
        scale = 2e-2
        ori_noise = np.random.normal(loc=0.0, scale=scale, size=(3,))
        camera_extrinsic[:3, 3] += pos_noise
        camera_extrinsic[:3, :3] = R.from_euler("xyz", ori_noise, degrees=False).as_matrix() @ camera_extrinsic[:3, :3]
        self.set_camera_extrinsic(self.camera_name, camera_extrinsic, mujoco_format=True)

    def randomize_all_color(self):
        self.model.geom_rgba[:, :3] *= np.random.uniform(
            0.95, 1.05, (self.model.geom_rgba.shape[0], 3)
        )

    def randomize_background_color(self):
       
        geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.model.ngeom)]

        self.wall_geom_ids = []
        self.table_geom_ids = []
        for name in geom_names:
            if name is None:
                continue
            if "wall" in name:
                self.wall_geom_ids += [
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                    )
                ]
            if "table" in name:
                self.table_geom_ids += [
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                    )
                ]
        geom_ids = self.wall_geom_ids + self.table_geom_ids

        # full color randomization
        # self.model.geom_rgba[geom_ids, :3] = np.random.uniform(
        #     0, 1, (len(geom_ids), 3)
        # )
        # color jitter
        self.model.geom_rgba[geom_ids, :3] *= np.random.uniform(
            0.7, 1.3, (len(geom_ids), 3)
        )
        
    def randomize_light(self):
        
        # change light position and direction -> low impact
        light_names = ["light_left", "light_right"]
        for light_name in light_names:
            light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, light_name)
            # Adjust position and direction slightly
            scale = 3e-1
            self.model.light_pos[light_id] += np.random.normal(0.0, scale, size=3)
            self.model.light_dir[light_id] += np.random.normal(0.0, scale, size=3)
        
        self.model.light_castshadow = np.random.choice([0, 1])

        # change light color -> large impact
        scale = 3e-2
        self.model.light_ambient += np.random.normal(0.0, scale, size=3)
        self.model.light_diffuse += np.random.normal(0.0, scale, size=3)
        self.model.light_specular += np.random.normal(0.0, scale, size=3)

    def randomize(self):

        # reset to intial values
        self.reset_randomize()

        # randomize camera, background color, light
        self.randomize_camera_pose()
        self.randomize_background_color()
        # self.randomize_all_color()
        self.randomize_light()

        # push changes from model to data | reset mujoco data
        mujoco.mj_resetData(self.model, self.data)


class CubeEnv(RobotEnv):
    def __init__(
        self,
        xml_path,
        num_objs=1,
        size=0.03,
        obj_pos_dist=[[0.4, -0.1, 0.03], [0.6, 0.1, 0.03]],
        obj_ori_dist=[[0, 0], [0, 0], [-np.pi / 4, np.pi / 4]],
        obj_color_dist=None,
        # obj_color_dist=[[0, 0, 0], [1, 1, 1]],
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
        # self.obj_color_dist = obj_color_dist
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

        def sample_positions(N, box_min, box_max, d):
            for _ in range(10000):  # brute-force retries
                candidates = np.random.uniform(box_min, box_max, size=(N * 5, 2))
                selected = [candidates[0]]
                if N == 1:
                    return selected
                
                for pt in candidates[1:]:
                    if all(np.linalg.norm(pt - s) >= d for s in selected):
                        selected.append(pt)
                        if len(selected) == N:
                            return selected
        box_poss = sample_positions(self.num_objs, self.obj_pos_dist[0][:2], self.obj_pos_dist[1][:2], d=0.12)
        # box_poss = sample_positions(self.num_objs, self.obj_pos_dist[0][:2], self.obj_pos_dist[1][:2], d=0.06)
        
        for _ in range(self.num_objs):
            # box_pos = np.random.uniform(self.obj_pos_dist[0], self.obj_pos_dist[1])
            box_pos = np.concatenate((box_poss[_], [0.03]))

            # ensure box is not too close to prev boxes
            # if len(obj_poses) > 0:
            #     solution_found = False
            #     for _ in range(100):
            #         box_pos = np.random.uniform(self.obj_pos_dist[0], self.obj_pos_dist[1])
            #         if np.all(np.abs(box_pos[:2] - np.array(obj_poses)[:,:2]) > 0.06):
            #             solution_found = True
            #             break
            #     if not solution_found:
            #         print("No solution found in 100 attempts, hard env reset - try to increase sample space")
            #         self.reset()
            box_euler = np.zeros(3)
            box_euler[2] = np.random.uniform(
                self.obj_ori_dist[2][0], self.obj_ori_dist[2][1]
            )
            try:
                box_quat = R.from_euler("xyz", box_euler, degrees=False).as_quat(
                    scalar_first=True
                )
            except:
                # old scipy version defaults to scalar_first=False
                box_quat = R.from_euler("xyz", box_euler, degrees=False).as_quat()
                box_quat = np.array([box_quat[3], box_quat[0], box_quat[1], box_quat[2]])
            obj_poses.append(np.concatenate((box_pos, box_quat)))
        obj_poses = np.concatenate(obj_poses, axis=0)
        self.set_obj_poses(obj_poses)

        main_colors = {
            "blue":    [0.0, 0.0, 1.0],
            "red":     [1.0, 0.0, 0.0],
            "green":   [0.0, 1.0, 0.0],
            "yellow":  [1.0, 1.0, 0.0],
            # "magenta": [1.0, 0.0, 1.0],
            # "orange":  [1.0, 0.5, 0.0]
        }
        colors = np.concatenate([main_colors[i] for i in np.random.choice(list(main_colors.keys()), size=self.num_objs, replace=False)], axis=0)
        
        if self.num_objs == 1:
            colors = main_colors["blue"]
        if self.num_objs == 2:
            colors = np.concatenate([main_colors["blue"], main_colors["red"]], axis=0)
        self.set_obj_colors(colors)

    def set_obj_poses(self, obj_poses):
        for i, obj_qpos_id in enumerate(self.obj_qpos_ids):
            obj_pose = obj_poses[i*7:(i+1)*7]
            self.data.qpos[
                self.model.jnt_qposadr[obj_qpos_id] : self.model.jnt_qposadr[
                    obj_qpos_id
                ]
                + 7
            ] = np.hstack(obj_pose)
        mujoco.mj_forward(self.model, self.data)

    def set_obj_colors(self, obj_colors):
        for i, obj_geom_id in enumerate(self.obj_geom_ids):
            self.model.geom_rgba[obj_geom_id] = np.concatenate((obj_colors[i*3:(i+1)*3], [1.0]))
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
        return np.concatenate(obj_poses, axis=0, dtype=np.float32)

    def get_obj_colors(self):
        obj_colors = []
        for obj_geom_id in self.obj_geom_ids:
            obj_colors.append(self.model.geom_rgba[obj_geom_id][:3])
        return np.concatenate(obj_colors, axis=0, dtype=np.float32)

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
            return self.get_obj_poses()[2] > 0.1
        elif task == "pick_and_place":
            return np.sum(np.abs(self.get_obj_poses()[:2] - self.get_obj_poses()[7:9])) < 0.06
        else:
            raise ValueError(f"Invalid task: {task}")