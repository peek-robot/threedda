# mujoco envs based on https://github.com/kevinzakka/mjctrl

import os
os.environ["MUJOCO_GL"] = "egl"
import mujoco
import imageio
from tqdm import trange
import mujoco.viewer
import numpy as np
import time
import robotic as ry
import matplotlib.pyplot as plt


def plan(C, grasp_direction="x"):
    ways = ry.KOMO_ManipulationHelper()
    ways.setup_sequence(C, 2, 1e-2, 1e-1, False, False, False)
    ways.grasp_box(1.0, "l_gripper", "obj", "l_palm", grasp_direction, 0.02)
    ways.komo.addObjective(
        [2.0],
        ry.FS.position,
        ["l_gripper"],
        ry.OT.eq,
        scale=[0, 0, 1e0],
        target=[0, 0, 0.3],
    )
    ret = ways.solve(0)
    X = ways.komo.getPath()
    # ways.view(True)
    if not ret.feasible:
        return None, None

    motion1 = ways.sub_motion(0)
    motion1.approach([0.8, 1.0], "l_gripper")
    ret = motion1.solve(0)
    # motion1.view(True)
    if not ret.feasible:
        return None, None

    motion2 = ways.sub_motion(1)
    ret = motion2.solve(0)
    # motion2.view(True)
    if not ret.feasible:
        return None, None

    return motion1.path, motion2.path

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # setup a configuration:
    C = ry.Config()
    C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaSingle.g"))
    # C.getFrame('l_panda_finger_joint1').setAttribute('motorKp', 100)
    # C.getFrame('l_panda_finger_joint1').setAttribute('motorKd', 10)

    table = C.getFrame("table")
    table.setShape(ry.ST.ssBox, [2.5, 2.5, 0.02, 0.005])
    table.setPosition([0, 0, -0.01])
    table.setQuaternion([1, 0, 0, 0])

    panda = C.getFrame("l_panda_base")
    panda.setPosition([0, 0, 0])
    panda.setQuaternion([1, 0, 0, 0])

    obj = C.addFrame("obj")
    obj.setPosition([-0.25, 0.1, 0.7]).setShape(
        ry.ST.ssBox, [0.06, 0.06, 0.06, 0.005]
    ).setColor([1, 0.5, 0]).setMass(0.1).setContact(True)

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path(
        "/home/memmelma/Projects/mjctrl/franka_emika_panda/scene.xml"
    )
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    gripper_names = ["joint8"]
    gripper_ids = np.array([model.actuator(name).id for name in gripper_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id

    # Camera settings
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.3, 0, 0.35]
    cam.distance = 1.5
    cam.azimuth = 120
    cam.elevation = -20

    width, height = 480, 480
    renderer = mujoco.Renderer(model, height=height, width=width)

    for i in trange(10, desc="Trajectories"):

        # sample random box pose
        box_pos = np.random.uniform([0.3, -0.3, 0.03], [0.7, 0.3, 0.03])
        box_quat = np.random.uniform([-1.0, 0.0, 0.0, -1.0], [1.0, 0.0, 0.0, 1.0])
        obj.setPosition(box_pos)
        obj.setQuaternion(box_quat)

        # plan the motion
        path1, path2 = plan(C, grasp_direction=np.random.choice(["x", "y"]))

        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Update "cube" joint pose in mujoco with box_pos, box_quat
        box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        data.qpos[box_joint_id : box_joint_id + 7] = np.hstack((box_pos, box_quat))
        mujoco.mj_forward(model, data)

        def execute(path, gripper=255.):
            frames = []
            for q in path:

                # np.clip(q, *[m[:7] for m in model.jnt_range.T], out=q)

                # Set the control signal and step the simulation.
                data.ctrl[actuator_ids] = q[dof_ids]
                # set gripper action
                data.ctrl[gripper_ids] = [gripper]
                mujoco.mj_step(model, data, nstep=50)

                # render
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
            return frames

        frames = []

        imgs = execute(path1, gripper=255.)
        frames.extend(imgs)
        
        imgs = execute(path2, gripper=0.)
        frames.extend(imgs)
        
        imageio.mimsave(f"gifs/path_{i}.gif", frames, fps=30)


if __name__ == "__main__":
    main()
