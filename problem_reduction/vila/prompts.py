import numpy as np

def get_prompt(quest, prompt_type, history=None, prompt_eval=False):

    if prompt_type == "path_gri":
        prompt = f"<image>\nIn the image, please execute the command described in <quest>{quest}</quest>.\nProvide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\nFormat your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\nThe tuple denotes point x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action.\nThe coordinates should be floats ranging between 0 and 1, indicating the relative locations of the points in the image.\n"
    elif prompt_type == "path":
        # prompt = f"<image>\nIn the image, please execute the command described in <quest>{quest}</quest>.\nProvide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\nFormat your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n<ans>[(25, 32), (32, 17), (13, 24), <action>Open Gripper</action>, (74, 21), <action>Close Gripper</action>, ...]</ans>\nThe tuple denotes point x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action.\nThe coordinates should be integers ranging between 0 and 100, indicating the relative locations of the points in the image.\n"
        prompt = f"<image>\nIn the image, please execute the command described in <quest>{quest}</quest>.\nProvide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\nFormat your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans>\nThe tuple denotes point x and y location of the end effector of the gripper in the image.\nThe coordinates should be integers ranging between 0.0 and 1.0, indicating the relative locations of the points in the image.\n"
    elif prompt_type == "mask":
        prompt = f"<image>\nIn the image, please execute the command described in <quest>{quest}</quest>.\nProvide a set of points denoting the areas the robot should mask to achieve the goal.\nFormat your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans>\nThe tuple denotes point x and y location of a mask in the image. \nThe coordinates should be integers ranging between 0.0 and 1.0, indicating the relative locations of the points in the image.\n"
    elif prompt_type == "path_mask":                   
        prompt = f"<image>\nIn the image, please execute the command described in <quest>{quest}</quest>.\nProvide a sequence of points denoting the trajectory of a robot gripper and a set of points denoting the areas the robot must see to achieve the goal.\nFormat your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\nTRAJECTORY: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans> MASK: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans>\nThe tuple denotes point x and y location of the end effector of the gripper in the image.\nThe coordinates should be integers ranging between 0.0 and 1.0, indicating the relative locations of the points in the image.\n"
    elif prompt_type == "path_mask_lang":                   
        prompt = f"<image>\nIn the image, please execute the command described in <quest>{quest}</quest>.\nProvide a sub-task description and a sequence of points denoting the trajectory of a robot gripper and a set of points denoting the areas the robot must see to achieve the goal.\nFormat your answer as a language instruction enclosed by <lang> and </lang> tags, and two lists of tuples enclosed by <ans> and </ans> tags. For example:\nINSTRUCTION: <lang>Pick up the coffee mug on the right.</lang> TRAJECTORY: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans> MASK: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans>\nThe tuple denotes point x and y location of the end effector of the gripper in the image.\nThe coordinates should be integers ranging between 0.0 and 1.0, indicating the relative locations of the points in the image.\n"
    elif prompt_type == "path_mask_history":
        prompt = f"<image>\nIn the image, please execute the command described in <quest>{quest}</quest>.\nThe previous sequence of points denoting the trajectory of a robot gripper is {str(history)}\nProvide a sequence of points denoting the trajectory of a robot gripper and a set of points denoting the areas the robot must see to achieve the goal.\nFormat your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\nTRAJECTORY: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans> MASK: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans>\nThe tuple denotes point x and y location of the end effector of the gripper in the image.\nThe coordinates should be integers ranging between 0.0 and 1.0, indicating the relative locations of the points in the image.\n"
    elif prompt_type == "path_mask_history_lang":                   
        prompt = f"<image>\nIn the image, please execute the command described in <quest>{quest}</quest>.\nThe previous sequence of points denoting the trajectory of a robot gripper is {str(history)}\nProvide a sub-task description and a sequence of points denoting the trajectory of a robot gripper and a set of points denoting the areas the robot must see to achieve the goal.\nFormat your answer as a language instruction enclosed by <lang> and </lang> tags, and two lists of tuples enclosed by <ans> and </ans> tags. For example:\nLANGUAGE: <lang>Pick up the coffee mug on the right.</lang> TRAJECTORY: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans> MASK: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), (0.74, 0.21), ...]</ans>\nThe tuple denotes point x and y location of the end effector of the gripper in the image.\nThe coordinates should be integers ranging between 0.0 and 1.0, indicating the relative locations of the points in the image.\n"
    elif prompt_type == "robotpoint":
        prompt = f"Identify some points in the free space on the {quest}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
    elif prompt_type == "hamster":
        # https://github.com/liyi14/HAMSTER_beta/blob/526a37f59f97c445005fcdf28f2cfb81ea742e4b/gradio_server_example.py#L159
        prompt = f"\nIn the image, please execute the command described in <quest>{quest}</quest>.\n"
        "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
        "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
        "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\n"
        "The tuple denotes point x and y location of the end effector in the image. The action tags indicate gripper actions.\n"
        "Coordinates should be floats between 0 and 1, representing relative positions.\n"
        "Remember to provide points between <ans> and </ans> tags and think step by step."
    
    else:
        raise NotImplementedError(f"Prompt type {prompt_type} not implemented.")

    if prompt_eval:
        return prompt.replace("<image>\n", "")
    return prompt

def get_answer_from_path(path, round_decimals=2):
    path = np.asarray(path)

    # construct answer like "<ans>[(73, 68), <action>Close Gripper</action>, (56, 59), <action>Open Gripper</action>]</ans>"
    answer = "<ans>["
    for i in range(len(path)):
        point = np.around(path[i], round_decimals)
        answer += f"({point[0]}, {point[1]}), "
    
    # remove last comma
    answer = answer[:-2]
    answer += "]</ans>"

    return answer

def get_answer_from_lang(lang):
    return "<lang>" + lang + "</lang>"
