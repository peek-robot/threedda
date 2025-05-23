# mujoco envs based on https://github.com/kevinzakka/mjctrl

import os
import glob
import re
os.environ["MUJOCO_GL"] = "egl"
import mujoco
import imageio
from tqdm import trange
import mujoco.viewer
import numpy as np
import time
import robotic as ry
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET

def add_objects_to_mujoco_xml(xml_file, num_objs=3, mass=0.05, size=0.03, colors=None):

    # Parse XML from string
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Locate worldbody
    worldbody = root.find("worldbody")

    # Add new cubes dynamically
    for i in range(num_objs):
        cube = ET.Element("body", name=f"cube_{i}", pos="1.0 1.0 0.03")
        ET.SubElement(cube, "inertial", pos="0 0 0", mass=f"{mass}", diaginertia="0.0002 0.0002 0.0002")
        ET.SubElement(cube, "freejoint", name=f"cube_{i}")
        color = colors[i] if colors is not None else np.random.uniform([0, 0, 0], [1, 1, 1])
        color_str = " ".join(map(str, color.tolist() + [1.0]))
        # import IPython; IPython.embed()
        ET.SubElement(cube, "geom", name=f"cube_{i}", type="box", size=f"{size} {size} {size}",
                        contype="1", conaffinity="1", rgba=color_str)
        worldbody.append(cube)

    # Locate or create the keyframe section
    keyframe_section = root.find("keyframe")
    if keyframe_section is None:
        keyframe_section = ET.Element("keyframe")
        root.append(keyframe_section)

    # Add a new keyframe "home_tmp"
    key = ET.SubElement(keyframe_section, "key", name="home_tmp")
    qpos_values = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04] + [1.0, 1.0, 0.03, 0, 0, 0, 0] * num_objs
    ctrl_values = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255]
    key.set("qpos", " ".join(map(str, qpos_values)))
    key.set("ctrl", " ".join(map(str, ctrl_values)))

    # Convert modified XML back to a string
    modified_xml = ET.tostring(root, encoding="unicode")

    return modified_xml


def plan(C, obj_names, grasp_direction="x", visualize=False):

    if visualize:
        C.view(False)

    # stacking: cube_0 <- cube_1 <- cube_2
    # step 1: stack cube_1 on cube_0
    # step 2: stack cube_2 on cube_1
    # ...

    margin = 0.02
    paths = []

    # import IPython; IPython.embed()
    
    for i in range(1, len(obj_names)):

        timestep = 0
        ways = ry.KOMO_ManipulationHelper()

        ways.setup_pick_and_place_waypoints(C, "l_gripper", obj_names[i])

        # pick
        # # grasp_direction="x"
        timestep = timestep + 1
        ways.grasp_box(
            timestep, "l_gripper", obj_names[i], "l_palm", grasp_direction, margin
        )
        
        # place
        place_direction = "z"
        timestep = timestep + 1
        ways.place_box(
            timestep, obj_names[i], obj_names[i - 1], "l_palm", place_direction, margin
        )
        # make sure objs are aligned
        ways.komo.addObjective([timestep], feature=ry.FS.positionDiff, frames=[obj_names[i], obj_names[i - 1]], scale=[1., 1., 0.], type=ry.OT.eq)
        ways.komo.addObjective([timestep], feature=ry.FS.qQuaternionNorms, frames=[obj_names[i], obj_names[i - 1]], type=ry.OT.eq)
        # ways.komo.addObjective([timestep], feature=ry.FS.quaternionDiff, frames=[obj_names[i], obj_names[i - 1]], type=ry.OT.eq)
        
        ret = ways.solve(0)
        X = ways.komo.getPath()
        if not ret.feasible:
            print(
                f"No feasible path found [init]"
            )
            return None, None

        # solve pick motion
        motion1 = ways.sub_motion(0)
        motion1.retract([0.0, 0.2], "l_gripper")
        motion1.approach([0.8, 1.0], "l_gripper")

        ret = motion1.solve(0)
        if not ret.feasible:
            print(
                f"No feasible path found at timestep {timestep} [grasp {obj_names[i]}]"
            )
            return None, None
        paths.append(motion1.path)

        # place pick motion
        motion2 = ways.sub_motion(1)

        ret = motion2.solve(0)
        if not ret.feasible:
            print(
                f"No feasible path found at timestep {timestep} [place {obj_names[i]} on {obj_names[i-1]}"
            )
            return None, None
        paths.append(motion2.path)

        def play_no_render(motion, C: ry.Config):
            dofs = C.getJointIDs()
            path = motion.komo.getPath(dofs)
            for t in range(motion.path.shape[0]):
                C.setJointState(path[t])

        def play_save_png(motion, C: ry.Config):
            dofs = C.getJointIDs()
            path = motion.komo.getPath(dofs)
            for t in range(motion.path.shape[0]):
                C.setJointState(path[t])
                C.view(False, f'step {t}\n{motion.info}')
                # time.sleep(duration/motion.path.shape[0])
                C.get_viewer().savePng("/tmp/")

        # update kinematics
        if visualize:
            # motion1.play(C)
            play_save_png(motion1, C)
            C.attach("l_gripper", obj_names[i])
            # motion2.play(C)
            play_save_png(motion2, C)
            # NOTE: order matters, otherwise gripper - obj_names[i] joint will prevent futher picks
            C.attach(obj_names[i - 1], obj_names[i])
        else:
            play_no_render(motion1, C)
            C.attach("l_gripper", obj_names[i])
            play_no_render(motion2, C)
            C.attach(obj_names[i - 1], obj_names[i])
    
    for i in range(1, len(obj_names)):
        # HACK: connect objs to table -> breaks joint that connects to other objs
        C.attach("table", obj_names[i])

        # # C.delFrame(obj_names[i])
        # frame = C.getFrame(obj_names[i])
        # frame.setJoint("none")

    # collect all .pngs at path: format "gifs0050.png" -> sort by number
    # write code line by line, no functions

    if visualize:
        png_paths = sorted(glob.glob("/tmp/*.png"), key=lambda x: int(re.search(r"\d+", x).group()))
        frames = []
        for p in png_paths:
            frames.append(imageio.imread(p))
            os.remove(p)
    else:
        frames = None
        
    return paths, frames


# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    visualize = "robotic" # "mujoco"
    num_objs = 4
    mass=0.05
    size=0.03

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

    obj_names = [f"cube_{i}" for i in range(num_objs)]
    colors = np.random.uniform([0, 0, 0], [1, 1, 1], size=(len(obj_names), 3))
    
    # HACK: overwrite colors
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])

    for i, name in enumerate(obj_names):
        obj = C.addFrame(name)
        obj.setPosition([-0.25, 0.1, 0.7]).setShape(
            ry.ST.ssBox, [size*2, size*2, size*2, 0.005]
        ).setColor(colors[i]).setMass(mass).setContact(True)

    # if visualize == "robotic":
    #     C.view()

    # Load the model and data.
    # model = mujoco.MjModel.from_xml_path(
    #     "/home/memmelma/Projects/mjctrl/franka_emika_panda/scene.xml"
    # )

    root_dir = "/home/memmelma/Projects/mjctrl/franka_emika_panda"
    modified_xml = add_objects_to_mujoco_xml(os.path.join(root_dir, "scene.xml"), num_objs=num_objs, mass=mass, size=size, colors=colors)
    with open(os.path.join(root_dir, "tmp.xml"), "w") as f:
        f.write(modified_xml)

    model = mujoco.MjModel.from_xml_path(os.path.join(root_dir, "tmp.xml"))
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
    key_name = "home_tmp"
    key_id = model.key(key_name).id

    # Camera settings
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.3, 0, 0.35]
    cam.distance = 1.5
    cam.azimuth = 120
    cam.elevation = -20

    width, height = 480, 480

    for i in trange(25, desc="Trajectories"):

        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # sample random box poses
        for j, name in enumerate(obj_names):
            box_pos = np.random.uniform([0.3, -0.3, 0.03], [0.7, 0.3, 0.03])
            box_quat = np.random.uniform([-1.0, 0.0, 0.0, -1.0], [1.0, 0.0, 0.0, 1.0])
            print(name, box_pos)
            obj = C.getFrame(name)
            obj.setPosition(box_pos)
            obj.setQuaternion(box_quat)

            # Update "cube" joint pose in mujoco with box_pos, box_quat
            box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[
                model.jnt_qposadr[box_joint_id] : model.jnt_qposadr[box_joint_id] + 7
            ] = np.hstack((box_pos, box_quat))
            

        mujoco.mj_forward(model, data)

        # plan the motion
        
        paths, frames = plan(
            C,
            obj_names,
            grasp_direction=np.random.choice(["x", "y"]),
            visualize=True, # visualize == "robotic",
        )
        if paths is None:
            print(f"[{i}] Planning failed.")
            continue

        if frames is not None:
            imageio.mimsave(
                f"/home/memmelma/Projects/robotic/gifs/path_{i}_robotic.gif", frames, fps=30
            )
        
        visualize = "mujoco"
        if visualize == "mujoco":
            renderer = mujoco.Renderer(model, height=height, width=width)

        def execute(path, gripper=255.0):
            frames = []
            for q in path:
                
                data.ctrl[actuator_ids] = q[dof_ids]
                data.ctrl[gripper_ids] = [gripper]
                mujoco.mj_step(model, data, nstep=50)

                # render
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
            return frames

        if visualize == "mujoco":

            frames = []

            pick = True
            for path in paths:
                gripper = 255.0 if pick else 0.0
                pick = not pick
                imgs = execute(path, gripper=gripper)
                frames.extend(imgs)

            imageio.mimsave(
                f"/home/memmelma/Projects/robotic/gifs/path_{i}_mujoco.gif", frames, fps=30
            )

        # import IPython

        # IPython.embed()


if __name__ == "__main__":
    main()
