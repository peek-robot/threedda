import h5py
import os
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import imageio

from problem_reduction.threedda.text_embed import CLIPTextEmbedder
from problem_reduction.masking.groundedsam import GroundedSam2Tracker

def instruction_to_dino_instr(instruction):
    # split objects and add gripper
    objects = instruction.replace("put the ", "").split(" on the ") + ["gripper."]
    # add "a " prefix to each object
    objects = ["a " + o for o in objects]
    # separate objects with ". "
    dino_instr = ". ".join(objects)
    return dino_instr

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--path", type=str, default="data/pick_and_place_1000_3_objs_va_high_cam.hdf5"
    )
    args = parser.parse_args()

    tracker = GroundedSam2Tracker()

    # load data
    file = h5py.File(args.path, "a")["data"]

    for i, dk in tqdm(enumerate(file.keys())):

        try:

            # load data
            rgbs = file[dk]["obs"]["rgb"][:]
            depths = file[dk]["obs"]["depth"][:]
            lang_instr = file[dk]["obs"].attrs["lang_instr"]

            dino_instr = instruction_to_dino_instr(lang_instr)

            tracker.reset(init_frame=Image.fromarray(rgbs[0]), text=dino_instr)

            # track masks
            masks_list = []
            for rgb in rgbs:
                t, masks = tracker.step(Image.fromarray(rgb))
                masks_list.append(masks)

            # apply masks
            rgbs_masked = tracker.apply_masks_to_frames(rgbs, masks_list)
            depth_masked = tracker.apply_masks_to_frames(depths[...,None], masks_list)

            # save data
            file[dk]["obs"]["rgb"][:] = np.stack(rgbs_masked)
            file[dk]["obs"]["depth"][:] = np.stack(depth_masked)[...,0]

        except Exception as e:
            print(e)
            continue

        if i < 10:
            print(lang_instr)
            print(dino_instr)
            imageio.mimsave(f"{lang_instr}.mp4", rgbs_masked)

