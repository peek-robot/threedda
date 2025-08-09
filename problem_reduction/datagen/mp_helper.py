import torch
from problem_reduction.datagen.blocks import compute_grasp_pose
from typing import List
import numpy as np


def subsample_min_velocity(
    qpos: np.ndarray, min_velocity: float, req_indices: List[int]
):
    """
    Given a trajectory, subsample it to have at least min_velocity between each point.
    req_indices is a list of indices where the trajectory is required to be sampled, e.g., key frames like grasping
    """
    indices = [0]
    last_idx = 0
    for i in range(1, len(qpos)):
        if i in req_indices:
            indices.append(i)
            last_idx = i
            continue
        velocity = np.linalg.norm(qpos[i] - qpos[last_idx])
        if velocity >= min_velocity:
            indices.append(i)
            last_idx = i
    return indices


def plan_pick_motion(obj_pose, mp, qpos=None, ee_pose=None):

    # get object and grasp pose
    grasp_pos, grasp_quat = compute_grasp_pose(obj_pose[0], obj_pose[1])

    # constants
    gripper_offset = 0.107
    cube_offset = 0.06

    # define targets
    grasp_target = {
        "ee_pos": torch.from_numpy(grasp_pos).float().cuda()[None]
        + torch.tensor([[0, 0, gripper_offset]]).cuda(),
        "ee_quat": torch.from_numpy(grasp_quat).float().cuda()[None],
    }
    grasp_gripper = 1
    pre_grasp_target = {
        "ee_pos": grasp_target["ee_pos"] + torch.tensor([[0, 0, cube_offset]]).cuda(),
        "ee_quat": grasp_target["ee_quat"],
    }
    pre_grasp_gripper = 1
    post_grasp_target = {
        "ee_pos": grasp_target["ee_pos"] + torch.tensor([[0, 0, 0.2]]).cuda(),
        "ee_quat": grasp_target["ee_quat"],
    }
    post_grasp_gripper = 0

    # define start
    if ee_pose is not None:
        start = {
            "ee_pos": ee_pose[0],
            "ee_quat": ee_pose[1],
        }
    else:
        start = {"qpos": torch.from_numpy(qpos).float().cuda()[None]}

    # plan motions
    qpos_traj = []
    gripper_traj = []
    for gripper, target in zip(
        [pre_grasp_gripper, grasp_gripper, post_grasp_gripper],
        [pre_grasp_target, grasp_target, post_grasp_target],
    ):
        traj = mp.plan_motion(start, target)
        qpos_traj.append(traj.position.cpu().numpy())
        gripper_traj.append(np.ones(len(traj.position.cpu().numpy())) * gripper)
        start = {
            "qpos": torch.from_numpy(qpos_traj[-1][-1]).float().cuda()[None],
        }

    return qpos_traj, gripper_traj


def plan_pick_and_place_motion(obj_pose, place_pose, mp, qpos=None, ee_pose=None, cube_size=0.06):

    # get object and grasp pose
    grasp_pos, grasp_quat = compute_grasp_pose(obj_pose[0], obj_pose[1])
    place_pos, place_quat = compute_grasp_pose(place_pose[0], place_pose[1])

    # constants
    gripper_offset = 0.107

    # define targets
    grasp_target = {
        "ee_pos": torch.from_numpy(grasp_pos).float().cuda()[None]
        + torch.tensor([[0, 0, gripper_offset]]).cuda(),
        "ee_quat": torch.from_numpy(grasp_quat).float().cuda()[None],
    }
    grasp_gripper = 1
    pre_grasp_target = {
        "ee_pos": grasp_target["ee_pos"] + torch.tensor([[0, 0, cube_size]]).cuda(),
        "ee_quat": grasp_target["ee_quat"],
    }
    pre_grasp_gripper = 1

    place_target = {
        "ee_pos": torch.from_numpy(place_pos).float().cuda()[None]
        + torch.tensor([[0, 0, gripper_offset]]).cuda()
        + torch.tensor([[0, 0, cube_size]]).cuda(),
        "ee_quat": torch.from_numpy(place_quat).float().cuda()[None],
    }
    place_gripper = 0    
    pre_place_target = {
        "ee_pos": torch.from_numpy(place_pos).float().cuda()[None]
        + torch.tensor([[0, 0, gripper_offset]]).cuda()
        + torch.tensor([[0, 0, cube_size]]).cuda() * 2,
        "ee_quat": torch.from_numpy(place_quat).float().cuda()[None],
    }
    pre_place_gripper = 0
    # HACK: interpolate between place and pre-grasp target + add z offset to avoid collision
    place_grasp = (place_target["ee_pos"] + pre_grasp_target["ee_pos"]) / 2
    place_grasp[0, 2] += cube_size
    inter_place_target = {
        "ee_pos": place_grasp,
        "ee_quat": grasp_target["ee_quat"],
    }
    inter_place_gripper = 0
    post_place_target = {
        "ee_pos": place_target["ee_pos"] + torch.tensor([[0, 0, cube_size]]).cuda(),
        "ee_quat": grasp_target["ee_quat"],
    }
    post_place_gripper = 1

    # define start
    if ee_pose is not None:
        start = {
            "ee_pos": ee_pose[0],
            "ee_quat": ee_pose[1],
        }
    else:
        start = {"qpos": torch.from_numpy(qpos).float().cuda()[None]}

    # plan motions
    qpos_traj = []
    gripper_traj = []
    for gripper, target in zip(
        [pre_grasp_gripper, grasp_gripper, inter_place_gripper, pre_place_gripper, place_gripper, post_place_gripper],
        [pre_grasp_target, grasp_target, inter_place_target, pre_place_target, place_target, post_place_target],
    ):
        traj = mp.plan_motion(start, target)
        qpos_traj.append(traj.position.cpu().numpy())
        gripper_traj.append(np.ones(len(traj.position.cpu().numpy())) * gripper)
        start = {
            "qpos": torch.from_numpy(qpos_traj[-1][-1]).float().cuda()[None],
        }

    return qpos_traj, gripper_traj


def plan_motion(obj_pose, mp, qpos=None, ee_pose=None):

    obj_pos, obj_quat = obj_pose

    # plan pick motion
    grasp_pos, grasp_quat = compute_grasp_pose(obj_pos, obj_quat)

    target = {
        "ee_pos": torch.from_numpy(grasp_pos).float().cuda()[None]
        + torch.tensor([[0, 0, 0.107]]).cuda(),
        "ee_quat": torch.from_numpy(grasp_quat).float().cuda()[None],
    }

    if ee_pose is not None:
        start = {
            "ee_pos": ee_pose[0],
            "ee_quat": ee_pose[1],
        }
    else:
        start = {
            "qpos": (
                torch.from_numpy(qpos).float().cuda()[None]
                if qpos is not None
                else None
            ),
        }

    traj = mp.plan_motion(start, target)
    return traj.position.cpu().numpy()
