import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import LineString

def scale_path(path, min_in, max_in, min_out, max_out):
    return (path - min_in) / (max_in - min_in) * (max_out - min_out) + min_out

def smooth_path_rdp(points, tolerance=0.05):
    """
    Simplifies a line using the Ramer-Douglas-Peucker algorithm.

    :param points: List of (x, y) tuples representing the line points
    :param tolerance: The tolerance parameter to determine the degree of simplification
    :return: List of (x, y) tuples representing the simplified line points
    """
    if len(points[0])==2:
        line = LineString(points)
        simplified_line = line.simplify(tolerance, preserve_topology=False)
        return np.array(simplified_line.coords)
    else:
        raise NotImplementedError("Only 2D simplification is supported")
    

def project_point_cloud_to_2d(point_cloud, K, RT, return_depth=False):
    """
    Project a 3D point cloud back to a 2D image plane using camera intrinsics and extrinsics.
    Parameters:
    - point_cloud (np.array): 3D point cloud (N x 3).
    - K (np.array): Camera intrinsic matrix (3x3).
    - RT (np.array): Camera extrinsic matrix (3x4) combining rotation and translation.
    - return_depth (bool): Whether to return the depth values at the projected points.
    Returns:
    - points_2d (np.array): 2D pixel coordinates (N x 2).
    - (optional) depth (np.array): Depth values at the projected points (N x 1) if return_depth is True.
    """
    # Separate rotation (R) and translation (t) from the extrinsic matrix RT
    R = RT[:, :3]  # Rotation matrix (3x3)
    t = RT[:, 3]  # Translation vector (3x1)
    # Transform points from world space to camera space
    points_in_camera = R.T @ (point_cloud.T - t[:, np.newaxis])
    # Project the points onto the image plane
    projected_points = K @ points_in_camera
    # Normalize to get pixel coordinates
    u = projected_points[0] / projected_points[2]
    v = projected_points[1] / projected_points[2]
    depth = projected_points[2]
    # Round to nearest pixel and filter valid projections
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)
    if return_depth:
        return np.stack((u, v), axis=1), depth
    return np.stack((u, v), axis=1)

def generate_path_2d_from_obs(obs):
    """
    Generate 2D path in image plane from ee_pos.
    - obs (dict): dataset observation
    Returns:
    - paths (list): list of 2D paths for each camera
    """

    path = []
    points_3d = obs["ee_pos"]

    for pi, point in enumerate(points_3d):
        K = obs["camera_intrinsic"][pi]
        R = obs["camera_extrinsic"][pi, :3, :3]
        t = obs["camera_extrinsic"][pi, :3, 3]
        RT = np.hstack((R, t.reshape(-1, 1)))

        path.append(project_point_cloud_to_2d(point[None], K, RT))

    return np.concatenate(path)


def add_path_2d_to_img(img, path, cmap=None, color=None):
    """
    Add 2D path to image.
    - img (np.ndarray): image
    - path (np.ndarray): 2D path
    """

    if cmap is not None:
        norm = plt.Normalize(
            vmin=0, vmax=len(path) - 1
        )  # Normalize values to [0, len(path)-1]

    img_out = img.copy()
    for i in range(len(path) - 1):
        if cmap is not None:
            # Map the current index to a color in the colormap
            plt_cmap = getattr(plt.cm, cmap)
            color = plt_cmap(norm(i))[:3]  # Get RGB (ignore alpha channel)
            color = tuple(int(c * 255) for c in color)  # Convert to 8-bit color
        elif color is not None:
            color = color
        else:
            color = (255, 0, 0)

        # Draw the line with the computed color
        cv2.line(
            img_out,
            (int(path[i][0]), int(path[i][1])),
            (int(path[i + 1][0]), int(path[i + 1][1])),
            color,
            2,
        )

    return img_out