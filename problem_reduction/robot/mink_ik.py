from pathlib import Path
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import mink
from problem_reduction import ROOT_DIR


class MinkIK:
    """
    Inverse kinematics helper for MuJoCo + Mink on a Franka Panda model.

    Usage:
        ik = MinkIK(xml_path=..., ee_site_name="attachment_site")
        q = ik.solve(position, quaternion, dt=0.01)

    The pose must be in MuJoCo's convention: position (x, y, z) in meters and
    quaternion (w, x, y, z).
    """

    def __init__(
        self,
        xml_path: str | Path,
        ee_site_name: str = "attachment_site",
        target_mocap_name: str = "target",
        position_threshold: float = 1e-4,
        orientation_threshold: float = 1e-4,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 1e-2,
        solver: str = "quadprog",
        max_iterations: int = 20,
        lm_damping: float = 1.0,
        dt: float = 0.002,
    ) -> None:

        self.xml_path = Path(xml_path)

        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold

        self.position_cost = position_cost
        self.orientation_cost = orientation_cost
        self.posture_cost = posture_cost

        self.solver = solver
        self.max_iterations = max_iterations
        self.lm_damping = lm_damping
        self.dt = dt

        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path.as_posix())
        self.data = mujoco.MjData(self.model)

        self.ee_site_name = ee_site_name
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        self.target_mocap_name = target_mocap_name

        # Configuration and tasks
        self.configuration = mink.Configuration(self.model)
        self.end_effector_task = mink.FrameTask(
            frame_name=self.ee_site_name,
            frame_type="site",
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            lm_damping=self.lm_damping,
        )
        self.posture_task = mink.PostureTask(model=self.model, cost=self.posture_cost)
        self.tasks = [self.end_effector_task, self.posture_task]

        # Align mocap target with current end-effector pose
        mink.move_mocap_to_frame(
            self.model,
            self.data,
            self.target_mocap_name,
            self.ee_site_name,
            "site",
        )

    def _converge_ik(
        self, configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
    ):
        """
        Runs up to 'max_iters' of IK steps. Returns True if position and orientation
        are below thresholds, otherwise False.
        """
        for _ in range(max_iters):
            vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3)
            configuration.integrate_inplace(vel, dt)

            # Only checking the first FrameTask here (end_effector_task).
            # If you want to check multiple tasks, sum or combine their errors.
            err = tasks[0].compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            if pos_achieved and ori_achieved:
                return True
        return False

    def compute_fk(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for the given joint positions.
        """
        # Add gripper dim
        qpos = np.concatenate([qpos[:7], [0.04, 0.04]])

        self.data.qpos = qpos
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.site_xpos[self.ee_site_id].astype(np.float32)
        ee_quat = R.from_matrix(self.data.site_xmat[self.ee_site_id].reshape(3, 3)).as_quat(scalar_first=True)
        return np.concatenate([ee_pos, ee_quat])
    
    def compute_ik(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        q_init: np.ndarray,
    ) -> np.ndarray:
        """
        Solve IK to reach the provided end-effector pose.

        Args:
            position: (3,) world position in meters (MuJoCo convention)
            quaternion: (4,) world orientation quaternion [w, x, y, z]
            q_init: optional initial joint configuration to start from

        Returns:
            Joint positions as a NumPy array (same ordering/length as configuration.q)
        """

        # Add gripper dim
        q_init = np.concatenate([q_init[:7], [0.04, 0.04]])

        # # Keep posture task centered at the provided initial configuration
        self.configuration.update(q_init)
        self.posture_task.set_target_from_configuration(self.configuration)

        # Update mocap body to desired target pose (MuJoCo convention)
        self.data.mocap_pos[0] = np.asarray(position, dtype=float)
        self.data.mocap_quat[0] = np.asarray(quaternion, dtype=float)
        # mujoco.mj_forward(self.model, self.data)

        # Set the end-effector task target from the mocap body transform
        T_wt = mink.SE3.from_mocap_name(self.model, self.data, self.target_mocap_name)
        self.end_effector_task.set_target(T_wt)

        # Run IK
        success = self._converge_ik(
            configuration=self.configuration,
            tasks=self.tasks,
            dt=self.dt,
            solver=self.solver,
            pos_threshold=self.position_threshold,
            ori_threshold=self.orientation_threshold,
            max_iters=self.max_iterations,
        )
        # if not success:
        #     import IPython; IPython.embed()

        # assert success, "IK failed to converge"
        if not success:
            print("IK failed to converge")

        return self.configuration.q.copy()[:7]
