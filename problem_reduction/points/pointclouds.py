import numpy as np
import torch

def depth_to_points(depth, intrinsic, extrinsic, depth_scale=1000.0):
    height, width = depth.shape[:2]
    depth = depth.squeeze() / depth_scale
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsic[0, 2]) * (depth / intrinsic[0, 0])
    py = (py - intrinsic[1, 2]) * (depth / intrinsic[1, 1])
    points = np.stack((px, py, depth, np.ones(depth.shape)), axis=-1)
    points = (extrinsic @ points.reshape(-1, 4).T).T
    points = points[:, :3]
    return points

def zero_points(points, colors=None, crop_min=[-1.0, -1.0, -0.2], crop_max=[1.0, 1.0, 1.0]):
    crop_min = torch.tensor(crop_min, device=points.device).view(1, 1, 3)
    crop_max = torch.tensor(crop_max, device=points.device).view(1, 1, 3)

    mask_min = (points > crop_min).all(dim=-1)
    mask_max = (points < crop_max).all(dim=-1)
    valid_mask = mask_min & mask_max

    points[~valid_mask] = 0.
    if colors is not None:
        colors[~valid_mask] = 0.
    return points, colors

import json
def read_calibration_file(filename):

    with open(filename) as file:
        calib_file = json.load(file)

    calib_dict = {}
    for calib in calib_file:
        sn = calib["camera_serial_number"]
        calib_dict[sn] = {"intrinsic": {}, "extrinsic": {}}
        calib_dict[sn]["intrinsic"] = calib["intrinsics"]
        calib_dict[sn]["extrinsic"]["pos"] = calib["camera_base_pos"]
        calib_dict[sn]["extrinsic"]["ori"] = calib["camera_base_ori"]

    return calib_dict