import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

def add_objects_to_mujoco_xml(xml_file, num_objs=3, mass=0.05, size=0.03, colors=None):

    # Parse XML from string
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Locate worldbody
    worldbody = root.find("worldbody")

    # Add new cubes dynamically
    for i in range(num_objs):
        cube = ET.Element("body", name=f"cube_{i}", pos="1.0 1.0 0.03", quat="1 0 0 0")
        ET.SubElement(cube, "inertial", pos="0 0 0", mass=f"{mass}", diaginertia="0.0002 0.0002 0.0002")
        ET.SubElement(cube, "freejoint", name=f"cube_{i}")
        color = colors[i] if colors is not None else np.random.uniform([0, 0, 0], [1, 1, 1])
        color_str = " ".join(map(str, color.tolist() + [1.0]))
        # import IPython; IPython.embed()
        ET.SubElement(cube, "geom", name=f"cube_{i}", type="box", size=f"{size} {size} {size}", contype="1", conaffinity="1", rgba=color_str)
        # <site name="cube_0_orientation" type="ellipsoid" size="0.01 0.01 0.01" rgba="1 0 0 1" pos="0 0 0"/>
        ET.SubElement(cube, "site", name=f"cube_{i}_orientation", type="ellipsoid", size="0.01 0.01 0.01", rgba="1 0 0 1", pos="0 0 0")
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

def compute_grasp_pose(cube_pos, cube_quat):
    # Convert cube orientation to rotation matrix
    rot = R.from_quat(cube_quat, scalar_first=True)
    cube_rot_matrix = rot.as_matrix()

    # Cube's front (x-axis) and top (z-axis) in world coordinates
    front_vec = cube_rot_matrix[:, 0]  # x-axis
    grasp_z = -np.array([0, 0, 1])  # Top-down grasp

    # Make orthonormal basis: grasp_z, grasp_y, grasp_x
    grasp_y = np.cross(grasp_z, front_vec)
    if np.linalg.norm(grasp_y) < 1e-6:  # If front_vec aligned with z
        grasp_y = np.array([0, 1, 0])
    else:
        grasp_y /= np.linalg.norm(grasp_y)
    grasp_x = np.cross(grasp_y, grasp_z)

    # Construct grasp rotation matrix and convert to quaternion
    grasp_rot_matrix = np.column_stack((grasp_x, grasp_y, grasp_z))
    grasp_quat = R.from_matrix(grasp_rot_matrix).as_quat(scalar_first=True)

    return cube_pos, grasp_quat

def compute_all_grasp_poses(cube_pos, cube_quat):
    # Convert cube orientation to rotation matrix
    rot = R.from_quat(cube_quat, scalar_first=True)
    cube_rot_matrix = rot.as_matrix()

    # Cube's front (x-axis) and top (z-axis) in world coordinates
    front_vec = cube_rot_matrix[:, 0]  # x-axis
    grasp_z = -np.array([0, 0, 1])  # Top-down grasp

    # Normalize and check alignment
    grasp_y = np.cross(grasp_z, front_vec)
    if np.linalg.norm(grasp_y) < 1e-6:
        grasp_y = np.array([0, 1, 0])
    else:
        grasp_y /= np.linalg.norm(grasp_y)
    grasp_x = np.cross(grasp_y, grasp_z)

    # Base grasp rotation
    base_grasp_rot = np.column_stack((grasp_x, grasp_y, grasp_z))

    # Generate 4 grasp orientations by rotating about Z
    cube_poss, cube_quats = [], []
    for angle_deg in [0, 90, -90]:
        angle_rad = np.deg2rad(angle_deg)
        z_rot = R.from_euler('z', angle_rad).as_matrix()
        rot_matrix = base_grasp_rot @ z_rot
        grasp_quat = R.from_matrix(rot_matrix).as_quat(scalar_first=True)
        cube_poss.append(cube_pos)
        cube_quats.append(grasp_quat)

    return cube_poss, cube_quats