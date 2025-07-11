import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

def closest_points(path, points):
    """
    Find closest point in a path for each point in a list of points
    :param path: List/array of points defining the path
    :param points: List/array of points to find closest points for
    """
    path = np.asarray(path, dtype=float)  # Ensure correct format
    points = np.asarray(points, dtype=float)

    tree = cKDTree(path)  # Build a KD-tree for fast nearest-neighbor search
    distances, indices = tree.query(points)  # Find closest point in path for each point

    closest_pts = path[indices]  # Get actual coordinates of closest points
    return closest_pts, distances  # Return both points and distances

from shapely.geometry import LineString

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

def smooth_path(path, num_points=5):
    """
    Smooth a path using cubic interpolation and sample num_points points
    :param path: List of points defining the path
    :param num_points: Number of points to sample
    """
    path = np.asarray(path, dtype=float)
    t = np.linspace(0, 1, len(path))
    interp_x = interp1d(t, path[:, 0], kind='cubic')
    interp_y = interp1d(t, path[:, 1], kind='cubic')

    new_t = np.linspace(0, 1, num_points)
    smoothed_path = np.column_stack((interp_x(new_t), interp_y(new_t)))
    return smoothed_path.astype(int)

def scale_path(path, min_in, max_in, min_out, max_out):
    return (path - min_in) / (max_in - min_in) * (max_out - min_out) + min_out
    
def discretize_gripper_state(gri, threshold=0.1):
    """
    Discretize gripper state based on gradient and threshold
    :param gri: list of fripper states
    :param threshold: threshold for significant diff/gradient
    """

    # duplicate last element to account for diff/gradient operation
    gri = np.concatenate((gri.copy(), gri[-1:].copy()), axis=0)
    gradient = np.abs(np.gradient(gri))
    significant_change = gradient > threshold
    # 0 for closed, 1 for open
    discrete_state = (gri > np.mean(gri)).astype(int)

    # refine by marking transitions based on significant changes
    for i in range(1, len(gri)):
        if significant_change[i]:
            discrete_state[i:] = 1 if gri[i] > gri[i - 1] else 0
    
    return discrete_state
    
def compute_gripper_actions_from_path(path, gri, short_path=None, threshold=0.5):
    """
    Detect (un-)grasp points and (optional) return closest point in shorter path
    :param path: list of points defining the path
    :param short_path: short path to find closest points in
    """

    GRIPPER_OPEN = True
    gripper_open = gri < threshold
    open_points, close_points = [], []
    for i in range(len(path)):
        if gripper_open[i] and not GRIPPER_OPEN:
            point = path[i]
            # find closest point in short_path
            if short_path is not None:
                point, _ = closest_points(short_path, [path[i]])
            open_points.append(point.astype(int))
            GRIPPER_OPEN = True
        elif not gripper_open[i] and GRIPPER_OPEN:
            point = path[i]
            # find closest point in short_path
            if short_path is not None:
                point, _ = closest_points(short_path, [path[i]])
            close_points.append(point.astype(int))
            GRIPPER_OPEN = False
    
    return open_points, close_points


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

def generate_path_2d_from_obs(points_3d, camera_intrinsic, camera_extrinsic):
    """
    Generate 2D path in image plane from ee_pos.
    - obs (dict): dataset observation
    Returns:
    - paths (list): list of 2D paths for each camera
    """

    path = []

    for pi, point in enumerate(points_3d):
        K = camera_intrinsic[pi]
        R = camera_extrinsic[pi, :3, :3]
        t = camera_extrinsic[pi, :3, 3]
        RT = np.hstack((R, t.reshape(-1, 1)))

        path.append(project_point_cloud_to_2d(point[None], K, RT))

    return np.concatenate(path)