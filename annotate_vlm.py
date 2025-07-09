import h5py
import os
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/mmfs1/gscratch/weirdlab/memmelma/simvla/vila_utils/")
from scripts.inference_helpers import center_crop_and_resize, inference_vila, load_model, add_answer_to_img
from vila_utils.utils.prompts import get_prompt

if __name__ == "__main__":

    model_path = "memmelma/vila_3b_blocks_path_mask_fast" 
    path = "data/pick_and_place_1000_3_objs_va_high_cam.hdf5"

    # server_ip = "https://44ba-198-48-92-26.ngrok-free.app"
    server_ip = None
    
    # load data
    file = h5py.File(path, 'a')["data"]

    # load model
    base_name, base_path = None, None
    args_dict = {
        "model_path": model_path,
        "conv_mode": "vicuna_v1",
        "model_base": None,
        # "temperature": 0.2,
        # "top_p": None,
        # "num_beams": 1,
        "max_new_tokens": 512,
    }
    model_args = argparse.Namespace(**args_dict)
    if server_ip is None:
        model = load_model("vila", model_args)

    def vila_inference(rgb, lang_instr, prompt_type="path_mask", visualize=False):

        prompt = get_prompt(quest=lang_instr, prompt_type=prompt_type)
        image = Image.fromarray(rgb)

        # preprocess
        image = center_crop_and_resize(image, min(image.size), 384)
        message = [prompt, image]

        # inference
        answer_pred = inference_vila(message, model_args)

        # postprocess
        image = np.array(image)

        image, path_pred, mask_pred = add_answer_to_img(image, answer_pred, prompt_type, color="red", add_mask=True)

        return image, path_pred, mask_pred

    def vila_inference_api(server_ip, rgb, lang_instr, prompt_type="path_mask"):

        from client_vlm import send_request
        answer_pred = send_request(rgb, lang_instr, prompt_type, server_ip)
        
        image, path_pred, mask_pred = add_answer_to_img(rgb, answer_pred, prompt_type, color="red", add_mask=True)

        return image, path_pred, mask_pred

    for i, dk in tqdm(enumerate(file.keys())):

        # if "path_vlm" in file[dk]["obs"] and "mask_vlm" in file[dk]["obs"]:
        #     continue

        rgbs = file[dk]["obs"]["rgb"]
        lang_instr = file[dk]["obs"].attrs["lang_instr"]
        split_size = 15

        paths_pad, masks_pad, images = [], [], []
        # HACK: single sub-task for now
        for split_idx in range(1):
        # for split_idx in range(int(np.ceil(len(rgbs) / split_size))):
        

            # import IPython; IPython.embed()
            # import matplotlib.pyplot as plt
            # img = rgbs[split_idx*split_size]
            # plt.imsave(f"{lang_instr}.png", img)
            # from client_vlm import send_request
            # response = send_request(img, "left of the table", "robotpoint", server_ip)
            # response = "TRAJECTORY " + response

            # from vila_utils.utils.encode import scale_path
            # from vila_utils.utils.decode import get_path_from_answer
            # from scripts.inference_helpers import add_path_2d_to_img
            # path = get_path_from_answer(response, "path")
            # min_in, max_in = np.zeros(2), np.array([img.shape[1], img.shape[0]])
            # min_out, max_out = np.zeros(2), np.ones(2)
            # path = path[0] if len(path) == 2 else path
            # scaled_path = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
            # img = add_path_2d_to_img(img, scaled_path, line_size=3, circle_size=0, plot_lines=True, color="red")


            # plt.imsave(f"{lang_instr}_points.png", img)
            # import IPython; IPython.embed()


            # inference
            if server_ip is None:
                image, path, mask = vila_inference(rgbs[split_idx*split_size], lang_instr)
            else:
                image, path, mask = vila_inference_api(server_ip, rgbs[split_idx*split_size], lang_instr)
            images.append(image)

            # path shape (N, 2) pad N with zeros until N=P
            max_path_length = 255
            path_pad = np.pad(path, ((0, max_path_length - path.shape[0]), (0, 0)), mode="constant", constant_values=(-1.))
            # add batch dim (B, P, 2)
            path_pad = np.repeat(path_pad[None], rgbs[split_idx*split_size : (split_idx+1)*split_size].shape[0], axis=0)
            paths_pad.append(path_pad)

            # mask shape (N, 2) pad N with zeros until N=P
            max_mask_length = 255
            mask_pad = np.pad(mask, ((0, max_mask_length - mask.shape[0]), (0, 0)), mode="constant", constant_values=(-1.))
            # add batch dim (B, P, 2)
            mask_pad = np.repeat(mask_pad[None], rgbs[split_idx*split_size : (split_idx+1)*split_size].shape[0], axis=0)
            masks_pad.append(mask_pad)
        
        paths_pad = np.concatenate(paths_pad, axis=0)
        masks_pad = np.concatenate(masks_pad, axis=0)

        if i < 10:
            plt.imsave(f"{lang_instr}.png", np.concatenate(images, axis=1))

        # store raw path
        try:
            del file[dk]["obs"]["path_vlm"]
        except:
            pass
        file[dk]["obs"].create_dataset("path_vlm", data=paths_pad, dtype=np.float32)

        # store raw mask
        try:
            del file[dk]["obs"]["mask_vlm"]
        except:
            pass
        file[dk]["obs"].create_dataset("mask_vlm", data=masks_pad, dtype=np.float32)
