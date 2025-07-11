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
    
    def __init__(self, task, split, target, track_path, tfds_path, train_test_split=1., img_key='primary', seed=0):
        
        assert split in ['train', 'test'], f"Split {split} not found."
        assert target in ['path', 'mask', 'path_mask', 'path_mask_lang'], f"Target {target} not found."

        self.task = task
        self.movement_key = "movement_across_subtrajectory" if "processed" in self.task else "movement_across_video"

        self.split = split
        self.target = target

        self.f_track = h5py.File(track_path, 'r', swmr=True)[task]

        self.img_key_to_name = OXE_DATASET_CONFIGS[task]["image_obs_keys"]
        self.f_original = tfds.load(task, data_dir=tfds_path, split="train")
        self.f_original = tfds.as_numpy(self.f_original)
        self.ds_iter = iter(self.f_original)
        
        self.img_key = img_key

        self.curr_sample = None
        self.curr_sample_iter = None
        self.curr_timestep = None
        self.curr_idx = None
        self.curr_sample_len = None

        np.random.seed(0)
        train_keys = np.random.choice(list(self.f_track.keys()), int(len(self.f_track)*train_test_split), replace=False)
        test_keys = [k for k in self.f_track.keys() if k not in train_keys]

        self.demo_keys = train_keys if split == "train" else test_keys
        # HACK: sort demo_key [0, 1, 2, ...] for compatibility w/ step_dataset
        self.demo_keys = sorted(self.demo_keys, key=lambda x: int(x.split('_')[1]))

        self.step(self.get_indices()[0])

    def get_indices(self):
        return np.arange(0, len(self.demo_keys))
        sub_idcs = []
        # for all demo keys
        for idx in np.arange(0, len(self.demo_keys)):
            if self.img_key in self.f_track[self.demo_keys[idx]].keys():
                n_sub_idcs = len(self.f_track[self.demo_keys[idx]][self.img_key]["traj_splits_indices"])
                # for all sub-trajectories in demo key
                for sub_idx in np.arange(1, n_sub_idcs, 1):
                    sub_idcs.append(f"{idx}_{sub_idx}")
        return sub_idcs

    def step(self, idx):
        # if first, init iterator and step once
        if self.curr_idx is None:
            curr_sample = next(self.ds_iter)["steps"]
            self.curr_sample = curr_sample
            self.curr_sample_iter = iter(curr_sample)
            self.curr_sample_len = len(curr_sample)
            self.curr_idx = 0
        
        # just for safety
        if idx < self.curr_idx:
            raise ValueError(f"Cannot go backwards to index {idx} from {self.curr_idx}")

        # if idx is greater than curr_idx, step until idx is reached
        if idx > self.curr_idx:
            while idx > self.curr_idx:
                try:
                    curr_sample = next(self.ds_iter)["steps"]
                    self.curr_sample = curr_sample
                    self.curr_sample_iter = iter(curr_sample)
                    self.curr_sample_len = len(curr_sample)
                    self.curr_idx += 1
                except StopIteration:
                    # Reset iterator if we reach the end
                    self.ds_iter = iter(self.f_original)
                    curr_sample = next(self.ds_iter)["steps"]
                    self.curr_sample = curr_sample
                    self.curr_sample_iter = iter(curr_sample)
                    self.curr_sample_len = len(curr_sample)
                    self.curr_idx += 1

    def _idx_to_sub_idx(self, idx):
        key = self.demo_keys[idx]
        lower = int(self.curr_sample_len*0.)
        upper = int(self.curr_sample_len*0.3)
        try:
            timestep_start = np.random.randint(lower, upper)
        except:
            timestep_start = 0
        return key, timestep_start, None, None

    def load_quest(self, idx):
        
        step = tfds.as_numpy(self.step_data) # tfds.as_numpy(next(self.curr_sample))
        try:
            lang = step["observation"]["natural_language_instruction"].decode("utf-8")
        except:
            lang = step["language_instruction"].decode("utf-8")
        
        return lang
        
    def load_path(self, idx):

        assert self.target == "path" or self.target == "path_mask" or self.target == "path_mask_lang"
        
        key, timestep_start, _, lang_idx = self._idx_to_sub_idx(idx)

        path = self.f_track[key][self.img_key]["gripper_positions"][timestep_start:]
        
        # HACK: tracking pipeline sometimes produces "idle" sub-trajectories with no path movement
        # compute distances between adjacent points and skip sub-trajectory if too small
        dists = np.sum(np.abs(path[1:] - path[:-1]))
        # in pixel space: move at least 20% in img space
        h, w = self.f_track[key][self.img_key]["masked_frames"].shape[-2:]
        if dists < 0.2*h:
            # len(path) == 1 will cause skip
            path = path[0]

        return path
    
    def load_lang(self, idx):

        key, timestep_start, _, lang_idx = self._idx_to_sub_idx(idx)
        # lang = self.f_track[key][self.img_key]["trajectory_labels"][lang_idx]

        # step = next(self.curr_sample)
        
        if self.target == "path_mask_lang":
            lang = self.f_track[key][self.img_key]["trajectory_labels"][lang_idx].decode('utf-8')
        else:
            lang = None

        return lang

    def load_mask(self, idx):

        assert self.target == "mask" or self.target == "path_mask" or self.target == "path_mask_lang"

        key, timestep_start, _, lang_idx = self._idx_to_sub_idx(idx)

        significant_points = self.f_track[key][self.img_key]["significant_points"]
        # movement_across_video = self.f_track[key]["primary"][self.movement_key]
        stopped_points = self.f_track[key][self.img_key]["stopped_points"]
        current_gripper_position = self.f_track[key][self.img_key]["gripper_positions"][timestep_start]
        all_points = np.concatenate([significant_points[timestep_start].astype(np.uint16), stopped_points[timestep_start].astype(np.uint16)])
        all_points = np.unique(all_points, axis=0)
        return all_points
        

    def save_image(self, idx, img_dir):
        
        key, timestep_start, _, lang_idx = self._idx_to_sub_idx(idx)

        # if img key doesn't exist, skip
        if self.img_key not in self.f_track[key].keys():
            return None, None

        mask_shape = self.f_track[key][self.img_key]["masked_frames"].shape[-2:]

        # load image from tfds dataset
        img = None
        for i, step in enumerate(self.curr_sample):
            self.step_data = step
            if i == timestep_start:
                img = tfds.as_numpy(step["observation"][self.img_key_to_name[self.img_key]]).astype(np.uint8)
                break

        # if img is empty, skip
        if img is None or img.sum() == 0:
            print("skipping", idx, timestep_start, _)
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
        # if exists, delete args.data_dir
        if os.path.exists(args.data_dir):
            import shutil
            shutil.rmtree(args.data_dir)

        args.save_sketches_every_n = 1
    else:
        args.data_dir = os.path.join(args.data_dir, f"oxe_processed_{args.target}")
        args.data_dir = args.data_dir + "_rw" if args.reword_quest else args.data_dir

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
        
        # copy track_path to /tmp/dataset_movement_and_masks.h5
        # generate random string
        tmp_track_path = f"/tmp/{dataset_name}/{args.img_key}/dataset_movement_and_masks.h5"
        import shutil
        # if doesn't exist, copy
        # # delete tmp_track_path if exists
        # if os.path.exists(tmp_track_path):
        #     os.remove(tmp_track_path)
        if not os.path.exists(tmp_track_path):
            os.makedirs(os.path.dirname(tmp_track_path), exist_ok=True)
            shutil.copy(track_path, tmp_track_path)
        track_path = tmp_track_path

        pbar.update(1)
        
        if i > 0 and args.debug:
            continue
        
        print("[CONVERT]", dataset_name)

        dataclass = TrackingDataclass(dataset_name, args.split, args.target, track_path=track_path, tfds_path=tfds_path, train_test_split=args.train_test_split, img_key=args.img_key)

        task = dataset_name + "_" + args.img_key + "_" + args.target
        task = task + "_rw" if args.reword_quest else task

        convert_dataset(task=task, dataclass=dataclass, split=args.split, data_dir=args.data_dir, prompt_type=args.target, reword_quest=args.reword_quest, num_samples=args.num_samples, path_rdp_tolerance=args.path_rdp_tolerance, mask_rdp_tolerance=args.mask_rdp_tolerance, save_sketches_every_n=args.save_sketches_every_n)
