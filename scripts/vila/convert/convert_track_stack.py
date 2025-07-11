import os
import json
import h5py
import argparse
import numpy as np
from PIL import Image

from vila_utils.utils.convert import convert_dataset

class TrackingDataclass():
    
    def __init__(self, task, split, target, data_path, num_points_mask=15, train_test_split=0.95, seed=0, img_key='primary', flip_img=False):
        
        assert split in ['train', 'test'], f"Split {split} not found."
        assert target in ['path', 'mask', 'path_mask'], f"Target {target} not found."

        self.task = task
        
        self.split = split
        self.target = target
        self.num_points_mask = num_points_mask

        self.img_key = img_key
        
        self.f_data = h5py.File(data_path, 'r', swmr=True)["data"]

        # seed train/test split
        self.seed = seed
        np.random.seed(self.seed)

        train_keys = np.random.choice(list(self.f_data.keys()), int(len(self.f_data)*train_test_split), replace=False)
        test_keys = [k for k in self.f_data.keys() if k not in train_keys]

        self.demo_keys = train_keys if split == "train" else test_keys

        self.demo_keys = sorted(self.demo_keys, key=lambda x: int(x.split('_')[1]))
    
    def get_indices(self):
        sub_idcs = []
        # for all demo keys
        for idx in np.arange(0, len(self.demo_keys)):
            if self.img_key in self.f_data[self.demo_keys[idx]]["obs"].keys():
                
                # HACK
                n_sub_idcs = 2
                
                # len(self.f_data[self.demo_keys[idx]][self.img_key]["traj_splits_indices"])
                # for all sub-trajectories in demo key
                for sub_idx in np.arange(0, n_sub_idcs-1):
                    sub_idcs.append(f"{idx}_{sub_idx}")

        return sub_idcs

    def step(self, idx):
        pass

    
    def _idx_to_sub_idx(self, idx):
        idx = [int(i) for i in idx.split("_")]
        key = self.demo_keys[idx[0]]

        # HACK
        traj_splits_indices = [0, -1]

        timestep_start = traj_splits_indices[idx[1]]
        timestep_end = traj_splits_indices[idx[1]+1]
        traj_len = timestep_end - timestep_start
        
        # HACK: since _idx_to_sub_idx is called multiple times for the same demo, make seed dependent on idx
        np.random.seed(self.seed + idx[0])

        # sample +/- 1/5 of the sub-trajectory
        margin = (timestep_end - timestep_start) // 5
        if margin > 0:
            timestep_start = timestep_start + np.random.randint(-margin, margin)
            timestep_end = timestep_end + np.random.randint(-margin, margin)

        # HACK: skip first 20% of first sub-traj
        timestep_start = max(timestep_start, int(traj_len*0.2))
        timestep_end = min(timestep_end, traj_splits_indices[-1])

        lang_idx = idx[1]

        return key, timestep_start, timestep_end, lang_idx

    def load_quest(self, idx):
        # key, timestep = self._index_to_key_idx(idx)
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)
        
        quest = self.f_data[key]["obs"].attrs["lang_instr"]
        
        return quest

    def load_path(self, idx):
    
        assert self.target == "path" or self.target == "path_mask"
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)

        path = self.f_data[key]["obs"]["path"][0][timestep_start:timestep_end]

        return path

    def load_mask(self, idx):

        assert self.target == "mask" or self.target == "path_mask"
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)

        mask = self.f_data[key]["obs"]["mask"][0][timestep_start:timestep_end]
        
        return mask


    def save_image(self, idx, img_dir):
        
        # key, timestep = self._index_to_key_idx(idx)
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)
        
        img = self.f_data[key]["obs"][self.img_key][timestep_start].astype(np.uint8)
        
        img_name = self.split + "_" + self.task + "_" + str(idx) + ".jpg"
        img_path = os.path.join(img_dir, img_name)

        os.makedirs(img_dir, exist_ok=True)
        Image.fromarray(img).save(img_path)
        
        return img, img_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to convert')
    parser.add_argument('--split', type=str, default='train', help='train or test split')
    parser.add_argument('--train_test_split', type=float, default=1.0, help='train (train_test_split) test (1-train_test_split) split')
    parser.add_argument('--img_key', type=str, default='rgb', help='img key to use, one of [primary, secondary, tertiary]')
    
    parser.add_argument('--target', type=str, default='path', help='path OR mask')
    parser.add_argument('--data_dir', type=str, default='/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/', help='data directory')
    parser.add_argument('--num_samples', type=int, default=np.inf, help='save num_samples')
    parser.add_argument('--num_seeds', type=int, default=1, help='number of seeds')
    parser.add_argument('--path_rdp_tolerance', type=float, default=0.05, help='tolerance for RDP path processing')
    parser.add_argument('--mask_rdp_tolerance', type=float, default=0.1, help='tolerance for RDP mask processing')
    parser.add_argument('--save_sketches_every_n', type=int, default=False, help='save every N-th sketch')
    parser.add_argument('--reword_quest', action="store_true", help='reword questions')
    parser.add_argument('--debug', action="store_true", help='debug mode')

    args = parser.parse_args()

    task = args.task
    data_path = f"/home/mmemmel/projects/vila/data/{args.task}.hdf5"

    if args.debug:
        args.data_dir = os.path.join(args.data_dir, "debug")
        # if exists, delete args.data_dir
        if os.path.exists(args.data_dir):
            import shutil
            shutil.rmtree(args.data_dir)

        args.save_sketches_every_n = 1
    else:
        args.data_dir = os.path.join(args.data_dir, f"stack_{args.target}_subtraj")


    # copy data_path to /tmp/dataset_tmp.h5
    tmp_path = f"/tmp/{task}/dataset_tmp.h5"
    import shutil
    # if doesn't exist, copy
    # if not os.path.exists(tmp_track_path):
    # remove tmp_track_path if exists
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    shutil.copy(data_path, tmp_path)
    data_path = tmp_path

    for seed in range(args.num_seeds):
        dataclass = TrackingDataclass(task, args.split, args.target, data_path=data_path, train_test_split=args.train_test_split, img_key=args.img_key, seed=seed)
        
        task_name = task + "_" + args.img_key + "_" + args.target + "_" + str(seed)
        convert_dataset(task=task_name, dataclass=dataclass, split=args.split, data_dir=args.data_dir, prompt_type=args.target, reword_quest=args.reword_quest, num_samples=args.num_samples, path_rdp_tolerance=args.path_rdp_tolerance, mask_rdp_tolerance=args.mask_rdp_tolerance, save_sketches_every_n=args.save_sketches_every_n)


# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_track_stack.py --task pick_and_place_10_3_objs_vlm --split train --target path_mask --debug
