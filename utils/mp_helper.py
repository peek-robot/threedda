import torch
from utils.blocks import compute_grasp_pose


def plan_pick_motion(obj_pose, mp, qpos=None, ee_pose=None):

    segments = []
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
        start = {"qpos": torch.from_numpy(qpos).float().cuda()[None]}

    traj = mp.plan_motion(start, target)
    segments.append(traj.position.cpu().numpy())

    # plan retract motion

    start = {
        # "ee_pos": target["ee_pos"],
        # "ee_quat": target["ee_quat"],
        "qpos": torch.from_numpy(segments[-1][-1])
        .float()
        .cuda()[None]
    }
    target = {
        "ee_pos": target["ee_pos"] + torch.tensor([[0, 0, 0.2]]).cuda(),
        "ee_quat": target["ee_quat"],
    }
    traj = mp.plan_motion(start, target)
    # first two steps equal to start
    segments.append(traj.position.cpu().numpy()[2:])

    return segments

def plan_pick_and_place_motion(obj_pose, place_pose, mp, qpos=None, ee_pose=None):

    segments = []
    obj_pos, obj_quat = obj_pose
    place_pos, place_quat = place_pose

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
        start = {"qpos": torch.from_numpy(qpos).float().cuda()[None]}

    traj = mp.plan_motion(start, target)
    segments.append(traj.position.cpu().numpy())

    # plan pre-place motion
    start = {
        # "ee_pos": target["ee_pos"],
        # "ee_quat": target["ee_quat"],
        "qpos": torch.from_numpy(segments[-1][-1])
        .float()
        .cuda()[None]
    }
    target = {
        "ee_pos": torch.from_numpy(place_pos).float().cuda()[None]
        + torch.tensor([[0, 0, 0.107 + 0.1]]).cuda(),
        "ee_quat": torch.from_numpy(grasp_quat).float().cuda()[None],
    }
    traj = mp.plan_motion(start, target)
    segments.append(traj.position.cpu().numpy())

    # plan retract motion
    start = {
        # "ee_pos": target["ee_pos"],
        # "ee_quat": target["ee_quat"],
        "qpos": torch.from_numpy(segments[-1][-1])
        .float()
        .cuda()[None]
    }
    target = {
        "ee_pos": torch.from_numpy(place_pos).float().cuda()[None]
        + torch.tensor([[0, 0, 0.107]]).cuda(),
        "ee_quat": torch.from_numpy(grasp_quat).float().cuda()[None],
    }
    traj = mp.plan_motion(start, target)
    segments.append(traj.position.cpu().numpy())

    return segments

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