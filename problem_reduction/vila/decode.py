import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

OPEN_GRIPPER = (1000, 1000)
CLOSE_GRIPPER = (1001, 1001)

def add_path_2d_to_img_alt_fast(
    image, points, line_size=1, circle_size=0, plot_lines=True
):
    img_out = image.copy()

    if np.all(points <= 1):
        points = points * image.shape[1]

    points = points.astype(int)
    path_len = len(points)

    # Generate gradient from dark red to bright red
    reds = np.linspace(64, 255, path_len).astype(int)
    colors = [tuple(int(r) for r in (r_val, 0, 0)) for r_val in reds]

    for i in range(path_len - 1):
        color = colors[i]
        if plot_lines:
            cv2.line(img_out, tuple(points[i]), tuple(points[i + 1]), color, line_size)
        if circle_size > 0:
            cv2.circle(
                img_out,
                tuple(points[i]),
                max(1, circle_size),
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

    # Draw last point
    if circle_size > 0:
        cv2.circle(
            img_out,
            tuple(points[-1]),
            max(1, circle_size),
            colors[-1],
            -1,
            lineType=cv2.LINE_AA,
        )

    return img_out


def get_path_from_answer(s, prompt_type="path"):
    """
    Remove tags from answers string, extract path, and replace gripper actions.
    - s (str): answer string
    """

    def get_points_from_ans(s):
        # remove tags
        s = re.sub(r"</?ans>", "", s)
        # replace gripper actions
        s = re.sub(r"<action>Open Gripper</action>", str(OPEN_GRIPPER), s)
        s = re.sub(r"<action>Close Gripper</action>", str(CLOSE_GRIPPER), s)

        # extract numbers (tuples) and boolean values, preserving order
        elements = re.findall(r'\((\d*\.?\d+), (\d*\.?\d+)\)', s)
        # string to floats to array
        return np.array([(float(elem[0]), float(elem[1])) for elem in elements])
    if prompt_type == "path" or prompt_type == "mask":
        return np.clip(get_points_from_ans(s), 0, 1)
    elif prompt_type == "path_mask" or prompt_type == "path_mask_history":
        split = s.replace("TRAJECTORY: ", "").split(" MASK:")
        # split = ["<ans>[(0.61, 0.50), (0.48, 0.45), (0.43, 0.39), (0.53, 0.39)]</ans>", "<ans>[(0.61, 0.50), (0.48, 0.45), (0.43, 0.39), (0.53, 0.39)]</ans>"]
        traj = get_points_from_ans(split[0])
        mask = get_points_from_ans(split[1])
        traj = np.clip(traj, 0, 1)
        mask = np.clip(mask, 0, 1)
        # augment the mask with the path
        mask = np.concatenate([mask, traj], axis=0)
        return [traj, mask]
    elif prompt_type == "path_mask_lang" or prompt_type == "path_mask_history_lang":

        split = s.replace("INSTRUCTION: ", "").split(" TRAJECTORY: ")
        lang = split[0].replace("<lang>", "").replace("</lang>", "")

        split = split[1].split(" MASK: ")
        traj = split[0]
        mask = split[1]

        traj = get_points_from_ans(traj)
        mask = get_points_from_ans(mask)
        traj = np.clip(traj, 0, 1)
        mask = np.clip(mask, 0, 1)
        return [lang, traj, mask]


def add_path_2d_to_img(img, path, cmap=None, color=None):
    """
    Add 2D path to image.
    - img (np.ndarray): image
    - path (np.ndarray): 2D path
    """

    # copy image
    img_out = img.copy()

    path_len = len(path)

    # setup color(-map)
    if cmap is not None:
        plt_cmap = getattr(plt.cm, cmap)
        norm = plt.Normalize(vmin=0, vmax=path_len - 1)
    elif color is None:
        color = (255, 0, 0)
    
    # plot path
    for i in range(path_len - 1):
        
        # get color
        if cmap is not None:
            color = plt_cmap(norm(i))[:3]
            color = tuple(int(c * 255) for c in color)

        cv2.line(
            img_out,
            (int(path[i][0]), int(path[i][1])),
            (int(path[i + 1][0]), int(path[i + 1][1])),
            color,
            2,
        )

    return img_out

def add_mask_2d_to_img(img, points, mask_pixels=25):
    img_zeros = np.zeros_like(img)
    for point in points:
        x, y = point.astype(int)
        y_minus, y_plus = int(max(0, y-mask_pixels)), int(min(img.shape[0], y+mask_pixels))
        x_minus, x_plus = int(max(0, x-mask_pixels)), int(min(img.shape[1], x+mask_pixels))
        # example for masking out a square
        img_zeros[y_minus:y_plus, x_minus:x_plus] = img[y_minus:y_plus, x_minus:x_plus]
    return img_zeros

def get_mask_2d(img, points, mask_pixels=25):
    img_zeros = np.zeros(img.shape[:2])
    for point in points:
        x, y = point.astype(int)
        y_minus, y_plus = int(max(0, y-mask_pixels)), int(min(img.shape[0], y+mask_pixels))
        x_minus, x_plus = int(max(0, x-mask_pixels)), int(min(img.shape[1], x+mask_pixels))
        # example for masking out a square
        img_zeros[y_minus:y_plus, x_minus:x_plus] = 1
    return img_zeros

def add_path_2d_to_img_alt_fast(
    image, points, line_size=1, circle_size=0, plot_lines=True
):
    img_out = image.copy()

    if np.all(points <= 1):
        points = points * image.shape[1]

    points = points.astype(int)
    path_len = len(points)

    # Generate gradient from dark red to bright red
    reds = np.linspace(64, 255, path_len).astype(int)
    colors = [tuple(int(r) for r in (r_val, 0, 0)) for r_val in reds]

    for i in range(path_len - 1):
        color = colors[i]
        if plot_lines:
            cv2.line(img_out, tuple(points[i]), tuple(points[i + 1]), color, line_size)
        if circle_size > 0:
            cv2.circle(
                img_out,
                tuple(points[i]),
                max(1, circle_size),
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

    # Draw last point
    if circle_size > 0:
        cv2.circle(
            img_out,
            tuple(points[-1]),
            max(1, circle_size),
            colors[-1],
            -1,
            lineType=cv2.LINE_AA,
        )

    return img_out