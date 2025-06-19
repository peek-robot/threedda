# cuRobo
import torch
import numpy as np
from curobo.geom.types import Sphere, Cuboid
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
    CudaRobotModelConfig,
)
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    
class CuroboWrapper:

    def __init__(
        self,
        device=None,
        robot_file="franka.yml",
        world_file="collision_table.yml",
        ik_solver=False,
        fk_solver=False,
        mp_solver=True,
        interpolation_dt=0.1,
        random_obstacle=False,
    ):
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if "cuda" not in device:
            raise ValueError("CUDA is required for motion planning")
        
        self.device = device
        self.tensor_args = TensorDeviceType(device=device)
        self.interpolation_dt = interpolation_dt

        # create configs
        data_dict_in = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        # increase velocity scale to 1.0
        data_dict_in["kinematics"]["cspace"]["velocity_scale"] = [1.0] * 10
        # import IPython; IPython.embed()
        robot_cfg = RobotConfig.from_dict(
            data_dict_in,
            self.tensor_args,
        )
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        
        # leads to IK failure, not sure why
        # add random obstacle to world -> more diverse trajectories
        if random_obstacle:
            pos = np.random.uniform([0.3, -0.2, 0.2], [0.7, 0.2, 0.3])
            # obstacle_0 = Sphere(
            #     name="obstacle_0", radius=2.6, pose=[*pos, 0.0, 0.0, 0.0, 0.0]
            # )
            obstacle_0 = Cuboid(
                # name="obstacle_0", dims=[0.1, 0.1, 0.1], pose=[*pos, 0.0, 0.0, 0.0, 0.0]
                name="obstacle_0", dims=[0.1, 0.1, 0.1], pose=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            )
            world_cfg.add_obstacle(obstacle_0)

        # MP
        if mp_solver:
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                world_cfg,
                self.tensor_args,

                interpolation_dt=self.interpolation_dt,
                trajopt_tsteps=34,
                grad_trajopt_iters=500,
                trajopt_dt=0.5,
                js_trajopt_dt=0.5,

                # Zoey params
                # trajopt_tsteps=50,
                # grad_trajopt_iters=500,
                # trajopt_dt=0.5,
                # js_trajopt_dt=0.5,
                # js_trajopt_tsteps=34,

                # interpolation_steps=10000,
                # rotation_threshold=0.01,
                # position_threshold=0.001,
                # num_ik_seeds=100,
                # num_trajopt_seeds=50,
                # collision_checker_type=CollisionCheckerType.PRIMITIVE,

                # evaluate_interpolated_trajectory=True,
                # velocity_scale=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5]
            )
            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen.warmup(enable_graph=True)

            self.retract_cfg = self.motion_gen.get_retract_config()

        # FK
        if fk_solver or mp_solver:
            self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        # IK
        if ik_solver or mp_solver:
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                world_cfg,
                rotation_threshold=5e-2, # 0.05,
                position_threshold=5e-3, # 0.005,
                num_seeds=250,
                self_collision_check=True,
                self_collision_opt=True,
                tensor_args=self.tensor_args,
                use_cuda_graph=True,
            )
            self.ik_solver = IKSolver(ik_config)

    def compute_fk(self, qpos):
        pos, quat, _, _, _, _, _ = self.kin_model.forward(qpos)
        return pos, quat
    
    def compute_ik(self, pos, quat):
        pose = Pose(pos, quat, normalize_rotation=False)
        result = self.ik_solver.solve_single(pose)
        qpos = result.solution[0]
        return qpos
    
    def plan_motion(self, start, target, return_ee_pose=False):

        if "ee_pos" in start and "ee_quat" in start:
            ee_pos, ee_quat = start["ee_pos"], start["ee_quat"]
            qpos = self.compute_ik(ee_pos, ee_quat)
        elif "qpos" in start:
            qpos = start["qpos"]
        else:
            raise ValueError("Invalid start state")
        
        if "qvel" in start:
            qvel = start["qvel"]
            start = JointState.from_numpy(
                joint_names=self.kin_model.joint_names,
                position=qpos,
                velocity=qvel,
                # self.tensor_args.to_device([qpos]), self.kin_model.joint_names
            )
        else:
            start = JointState.from_position(
                position=qpos,
                joint_names=self.kin_model.joint_names,
                # self.tensor_args.to_device([qpos]), self.kin_model.joint_names
            )

        if "ee_pos" in target and "ee_quat" in target:
            target_ee_pos, target_ee_quat = target["ee_pos"], target["ee_quat"]
        else:
            target_ee_pos, target_ee_quat = target
        goal = Pose(target_ee_pos, target_ee_quat, normalize_rotation=False)
        
        
        result = self.motion_gen.plan_single(
            start, goal, MotionGenPlanConfig(max_attempts=1)
        )

        traj = result.get_interpolated_plan()
        if traj is None:
            raise ValueError
        # print(
        #     f"Trajectory Generated: success {result.success.item()} | len {len(traj)} | optimized_dt {result.optimized_dt.item()}"
        # )
        
        # replace joint position with ee pose
        if return_ee_pose:
            traj = self.kin_model.get_state(traj.position)

        return traj
    
    def plan_motion_set(self, ee_pos, ee_quat, targets_ee_pos, targets_ee_quat, return_ee_pose=False):

        # start = Pose(
        #     self.tensor_args.to_device([ee_pose[:3]]),
        #     self.tensor_args.to_device([ee_pose[3:]]),
        #     normalize_rotation=False,
        # )

        # result = self.ik_solver.solve_single(start)
        # qpos = result.solution[0]
        
        qpos = self.compute_ik(ee_pos, ee_quat)

        start = JointState.from_position(
            qpos,
            self.kin_model.joint_names,
            # self.tensor_args.to_device([qpos]), self.kin_model.joint_names
        )

        goalset = Pose(targets_ee_pos, targets_ee_quat, normalize_rotation=False)

        result = self.motion_gen.plan_goalset(
            start, goalset, MotionGenPlanConfig(max_attempts=1)
        )

        traj = result.get_interpolated_plan()
        if traj is None:
            raise ValueError
        # print(
        #     f"Trajectory Generated: success {result.success.item()} | len {len(traj)} | optimized_dt {result.optimized_dt.item()}"
        # )
        
        # replace joint position with ee pose
        if return_ee_pose:
            traj = self.kin_model.get_state(traj.position)

        return traj