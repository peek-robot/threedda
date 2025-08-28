import h5py
import os
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from problem_reduction.vila.prompts import get_prompt
from problem_reduction.vila.inference_helpers import (
    load_model
)
from problem_reduction.vila.inference_hamster import hamster_inference_api

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "--model_path", type=str, default="memmelma/vila_3b_blocks_path_mask_fast"
        "--model_path", type=str, default="memmelma/vila_3b_path_mask_fast"
    )
    parser.add_argument(
        "--path", type=str, default="data/pick_and_place_1000_3_objs_va_high_cam.hdf5"
    )
    parser.add_argument(
        "--server_ip", type=str, default=None
    )  # https://44ba-198-48-92-26.ngrok-free.app
    args = parser.parse_args()

    # load data
    file = h5py.File(args.path, "a")["data"]

    # load model
    base_name, base_path = None, None
    args_dict = {
        "model_path": args.model_path,
        "conv_mode": "vicuna_v1",
        "model_base": None,
        # "temperature": 0.2,
        # "top_p": None,
        # "num_beams": 1,
        "max_new_tokens": 512,
    }
    model_args = argparse.Namespace(**args_dict)
    if args.server_ip is None:
        model = load_model("vila", model_args)

    for i, dk in tqdm(enumerate(file.keys())):

        # if "path_vlm" in file[dk]["obs"].keys() and "mask_vlm" in file[dk]["obs"].keys():
        #     if not np.all(file[dk]["obs"]["path_vlm"][:] == 0) and not np.all(file[dk]["obs"]["mask_vlm"][:] == 0):
        #         continue

        # if "path_vlm" in file[dk]["obs"] and "mask_vlm" in file[dk]["obs"]:
        #     continue

        rgbs = file[dk]["obs"]["rgb"]
        lang_instr = file[dk]["obs"].attrs["lang_instr"]

        _, _, images = [], [], []

        # inference
        try:
            assert args.server_ip is not None
            image = rgbs[0]
            image, path = hamster_inference_api(
                image, lang_instr, model_name=args.model_path, server_ip=args.server_ip, prompt_type="hamster"
            )
            path = np.array(path)

        except Exception as e:
            path = np.zeros((1, 3))
        images.append(image)

        # path shape (N, 2) pad N with zeros until N=P
        max_path_length = 255
        path_pad = np.pad(
            path,
            ((0, max_path_length - path.shape[0]), (0, 0)),
            mode="constant",
            constant_values=(-1.0),
        )

        # add batch dim (B, P, 2)
        path_pad = np.repeat(
            path_pad[None],
            rgbs.shape[0],
            axis=0,
        )

        if i < 10:
            plt.imsave(f"{lang_instr}.png", np.concatenate(images, axis=1))

        # store raw path
        try:
            del file[dk]["obs"]["path_vlm"]
        except:
            pass
        file[dk]["obs"].create_dataset("path_vlm", data=path_pad, dtype=np.float32)

