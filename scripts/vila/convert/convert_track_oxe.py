import os
import cv2
import glob
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from vila_utils.utils.convert import convert_dataset

import psutil
import os

class TrackingDataclass():
    
    def __init__(self, task, split, target, track_path, tfds_path, train_test_split=1., third="first", img_key='primary', seed=0):
        
        assert split in ['train', 'test'], f"Split {split} not found."
        assert target in ['path', 'mask', 'path_mask'], f"Target {target} not found."

        self.task = task
        self.movement_key = "movement_across_subtrajectory" if "processed" in self.task else "movement_across_video"

        self.split = split
        self.target = target

        self.f_track = h5py.File(track_path, 'r', swmr=True)[task]

        self.img_key_to_name = OXE_DATASET_CONFIGS[task]["image_obs_keys"]
        self.f_original = tfds.load(task, data_dir=tfds_path, split="train")
        self.f_original = tfds.as_numpy(self.f_original)
        self.f_original = iter(self.f_original)
        
        self.img_key = img_key
        self.third = third

        self.curr_sample = None
        self.curr_timestep = None
        self.curr_idx = 0
        self.step_dataset()

        np.random.seed(0)
        train_keys = np.random.choice(list(self.f_track.keys()), int(len(self.f_track)*train_test_split), replace=False)
        test_keys = [k for k in self.f_track.keys() if k not in train_keys]

        self.demo_keys = train_keys if split == "train" else test_keys
        # HACK: sort demo_key [0, 1, 2, ...] for compatibility w/ step_dataset
        self.demo_keys = sorted(self.demo_keys, key=lambda x: int(x.split('_')[1]))

    def step_dataset(self):
        
        # sample trajectory
        self.curr_sample = next(self.f_original)["steps"]

        # set sampling boundaries
        if self.third == "first":     
            lower = int(len(self.curr_sample)*0.)
            upper = int(len(self.curr_sample)*0.3)
        elif self.third == "second":
            lower = int(len(self.curr_sample)*0.3)
            upper = int(len(self.curr_sample)*0.6)
        elif self.third == "third":
            lower = int(len(self.curr_sample)*0.6)
            upper = int(len(self.curr_sample)*0.9)
        else:
            raise ValueError(f"Third {self.third} not found.")

        if upper > lower:
            self.curr_timestep = np.random.randint(lower, upper)
        else:
            self.curr_timestep = 0
        # print_memory_usage()
        
    def get_indices(self):
        return np.arange(0, len(self.demo_keys))

    # def _get_valid_img_key(self, open_file):
        # for key in ['primary', 'secondary', 'tertiary']:
        # HACK
        # for key in self.img_keys:
        #     if key in open_file.keys():
        #         return key

    def step(self, idx):
        # HACK: quest is always called last, so next(dataset) after done if not last
        if idx < self.get_indices()[-1]:
            self.step_dataset()
            self.curr_idx += 1

            # HACK: skip until tracking data exists
            while f"episode_{self.curr_idx}" != self.demo_keys[idx+1]:
                self.step_dataset()
                self.curr_idx += 1

    def load_quest(self, idx):
        key = self.demo_keys[idx]
        timestep = self.curr_timestep
        
        step = next(iter(self.curr_sample))
        try:
            lang = step["observation"]["natural_language_instruction"].decode("utf-8")
        except:
            lang = step["language_instruction"].decode("utf-8")
        
        return lang
        
    def load_path(self, idx):
    
        assert self.target == "path" or self.target == "path_mask"
        key = self.demo_keys[idx]
        timestep = self.curr_timestep

        return self.f_track[key][self.img_key]["gripper_positions"][timestep:]
        
    def load_mask(self, idx):

        assert self.target == "mask" or self.target == "path_mask"
        key = self.demo_keys[idx]
        timestep = self.curr_timestep

        significant_points = self.f_track[key][self.img_key]["significant_points"]
        # movement_across_video = self.f_track[key]["primary"][self.movement_key]
        stopped_points = self.f_track[key][self.img_key]["stopped_points"]
        all_points = np.concatenate([significant_points[timestep].astype(np.uint16), stopped_points[timestep].astype(np.uint16)])
        all_points = np.unique(all_points, axis=0)
        return all_points
        

    def save_image(self, idx, img_dir):
        
        key = self.demo_keys[idx]
        timestep = self.curr_timestep
        
        # if img key doesn't exist, skip
        if self.img_key not in self.f_track[key].keys():
            return None, None

        mask_shape = self.f_track[key][self.img_key]["masked_frames"].shape[-2:]

        # load image from tfds dataset
        for i, step in enumerate(self.curr_sample):
            if i == timestep:
                img = step["observation"][self.img_key_to_name[self.img_key]].astype(np.uint8)
                # print("image_0", step["observation"]["image_0"].sum(), "image_1", step["observation"]["image_1"].sum(), "image_2", step["observation"]["image_2"].sum(), "image_3", step["observation"]["image_3"].sum())
                break

        # if img is empty, skip
        if img.sum() == 0:
            return None, None

        # resize to make sure img is scaled corresponding to path/mask
        img = cv2.resize(img, mask_shape)

        img_name = self.split + "_" + self.task + "_" + str(idx) + ".jpg"
        img_path = os.path.join(img_dir, img_name)

        os.makedirs(img_dir, exist_ok=True)
        Image.fromarray(img).save(img_path)
        
        return img, img_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None, help='custom task')
    parser.add_argument('--split', type=str, default='train', help='train or test split')
    parser.add_argument('--train_test_split', type=float, default=1.0, help='train (train_test_split) test (1-train_test_split) split')
    parser.add_argument('--img_key', type=str, default='primary', help='img key to use, one of [primary, secondary, tertiary]')
    parser.add_argument('--third', type=str, default='first', help='third to use, one of [first, second, third]')
    parser.add_argument('--target', type=str, default='path', help='path OR mask')
    parser.add_argument('--data_dir', type=str, default='/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/', help='data directory')
    parser.add_argument('--num_samples', type=int, default=np.inf, help='save num_samples')
    parser.add_argument('--path_rdp_tolerance', type=float, default=0.05, help='tolerance for RDP path processing')
    parser.add_argument('--mask_rdp_tolerance', type=float, default=0.1, help='tolerance for RDP mask processing')
    parser.add_argument('--save_sketches_every_n', type=int, default=False, help='save every N-th sketch')
    parser.add_argument('--reword_quest', action="store_true", help='reword questions')
    parser.add_argument('--debug', action="store_true", help='debug mode')

    args = parser.parse_args()

    if args.debug:
        args.data_dir = os.path.join(args.data_dir, "debug")
        args.save_sketches_every_n = 1
    else:
        args.data_dir = os.path.join(args.data_dir, f"oxe_processed_{args.target}")

    import sys
    sys.path.append("/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/")
    from oxe_configs import OXE_DATASET_CONFIGS, DATASET_TRANSFORMS

    # prevent TFDS from taking up all GPU memory
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow_datasets as tfds
    
    tfds_path = "/home/jeszhang/data/tensorflow_datasets/openx_datasets/"
    dataset_names = [x.split()[0] for x in DATASET_TRANSFORMS]
    if args.task is not None:
        dataset_names = [args.task]

    pbar = tqdm(total=len(dataset_names))
    for i, dataset_name in enumerate(dataset_names):
        
        track_path = "/home/jeszhang/data/masked_vla_data/oxe_processed_subtrajectory/"
        track_dir = os.path.join(track_path, dataset_name)
        if not os.path.isdir(track_dir):
            print("[SKIP]", track_dir, "not a directory")
            continue
        track_path = os.path.join(track_dir, "dataset_movement_and_masks.h5")

        pbar.update(1)
        
        if i > 0 and args.debug:
            continue
        
        print("[CONVERT]", dataset_name)

        dataclass = TrackingDataclass(dataset_name, args.split, args.target, track_path=track_path, tfds_path=tfds_path, train_test_split=args.train_test_split, img_key=args.img_key, third=args.third)

        task = dataset_name + "_" + args.img_key + "_" + args.target + "_" + args.third
        task = task + "_rw" if args.reword_quest else task
        convert_dataset(task=task, dataclass=dataclass, split=args.split, data_dir=args.data_dir, prompt_type=args.target, reword_quest=args.reword_quest, num_samples=args.num_samples, path_rdp_tolerance=args.path_rdp_tolerance, mask_rdp_tolerance=args.mask_rdp_tolerance, save_sketches_every_n=args.save_sketches_every_n)
