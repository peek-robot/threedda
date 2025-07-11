import os
import json
import h5py
import argparse
import numpy as np
from PIL import Image

from vila_utils.utils.convert import convert_dataset

class TrackingDataclass():
    
    def __init__(self, task, split, target, track_path, original_path, start_idx=0, num_points_mask=15, train_test_split=0.95, seed=0, img_key='primary', flip_img=False, history=50):
        
        assert split in ['train', 'test'], f"Split {split} not found."
        assert target in ['path', 'mask', 'path_mask', 'path_mask_history'], f"Target {target} not found."

        self.task = task
        self.movement_key = "movement_across_subtrajectory" if "processed" in self.task else "movement_across_video"

        self.split = split
        self.target = target
        self.num_points_mask = num_points_mask

        self.img_key = img_key
        self.history = history
        self.flip_img = flip_img

        if task == "stack_platforms2":
            self.f_track = h5py.File(track_path, 'r', swmr=True)[self.task+"_roboverse"]
        else:
            self.f_track = h5py.File(track_path, 'r', swmr=True)[self.task]

        self.f_original = h5py.File(original_path, 'r', swmr=True)["data"]
        
        # seed train/test split
        self.seed = seed
        np.random.seed(self.seed)

        train_keys = np.random.choice(list(self.f_track.keys()), int(len(self.f_track)*train_test_split), replace=False)
        test_keys = [k for k in self.f_track.keys() if k not in train_keys]

        self.demo_keys = train_keys if split == "train" else test_keys

        self.start_idx = start_idx
        self.demo_keys = sorted(self.demo_keys, key=lambda x: int(x.split('_')[1]))
    
        # self.demo_lens = np.cumsum([len(self.f_original[key]["obs"]["world_camera_low_res_image"])-self.start_idx for key in self.demo_keys])

    # def get_indices(self):
    #     return np.arange(0, self.demo_lens[-1], self.stride)
    # def get_indices(self):
    #     return np.arange(0, len(self.demo_keys))
    def get_indices(self):
        sub_idcs = []
        # for all demo keys
        for idx in np.arange(0, len(self.demo_keys)):
            if self.img_key in self.f_track[self.demo_keys[idx]].keys():
                n_sub_idcs = len(self.f_track[self.demo_keys[idx]][self.img_key]["traj_splits_indices"])
                # for all sub-trajectories in demo key
                for sub_idx in np.arange(0, n_sub_idcs-1):
                    sub_idcs.append(f"{idx}_{sub_idx}")
        return sub_idcs

    def step(self, idx):
        pass

    
    def _idx_to_sub_idx(self, idx):
        idx = [int(i) for i in idx.split("_")]
        key = self.demo_keys[idx[0]]
        traj_splits_indices = self.f_track[key][self.img_key]["traj_splits_indices"]
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

    # def _index_to_key_idx(self, idx):
        
    #     key = self.demo_keys[idx]
    #     curr_sample_len = len(self.f_original[key]["actions"])
    #     if self.third == "first":     
    #         lower = int(curr_sample_len*0.)
    #         upper = int(curr_sample_len*0.3)
    #     elif self.third == "second":
    #         lower = int(curr_sample_len*0.3)
    #         upper = int(curr_sample_len*0.6)
    #     elif self.third == "third":
    #         lower = int(curr_sample_len*0.6)
    #         upper = int(curr_sample_len*0.9)
    #     else:
    #         raise ValueError(f"Third {self.third} not found.")

    #     if upper > lower:
    #         timestep = np.random.randint(lower, upper)
    #     else:
    #         timestep = 0
    #     return key, timestep
        # # select task and demo
        # task_idx = np.searchsorted(self.demo_lens, idx + 1)
        # if task_idx == 0:
        #     timestep = idx
        # else:
        #     timestep = idx - self.demo_lens[task_idx - 1]
        # return self.demo_keys[task_idx], self.start_idx+timestep

    # def load_lang(self, idx):

    def load_quest(self, idx):
        # key, timestep = self._index_to_key_idx(idx)
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)
        if self.task == "stack_platforms2":
            return self.f_original[key]["obs"].attrs["task_instruction"]
        else:
            return json.loads(self.f_original.attrs["problem_info"])["language_instruction"]

    def load_path(self, idx):
    
        assert self.target == "path" or self.target == "path_mask" or self.target == "path_mask_history"
        # key, timestep = self._index_to_key_idx(idx)
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)

        path = self.f_track[key][self.img_key]["gripper_positions"][timestep_start:timestep_end]
        print("path", timestep_start, timestep_end)
        return path

    def load_path_history(self, idx):
    
        assert self.target == "path_mask_history"
        # key, timestep = self._index_to_key_idx(idx)
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)

        history = np.random.randint(20, 60)
        print("hist", max(0, timestep_start-history),timestep_start)
        path_history = self.f_track[key][self.img_key]["gripper_positions"][max(0, timestep_start-history):timestep_start]

        return path_history

    def load_mask(self, idx):

        assert self.target == "mask" or self.target == "path_mask" or self.target == "path_mask_history"
        # key, timestep = self._index_to_key_idx(idx)
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)

        significant_points = self.f_track[key][self.img_key]["significant_points"]
        # movement_across_video = self.f_track[key]["primary"][self.movement_key]
        stopped_points = self.f_track[key][self.img_key]["stopped_points"]
        all_points = np.concatenate([significant_points[timestep_start].astype(np.uint16), stopped_points[timestep_start].astype(np.uint16)])
        all_points = np.unique(all_points, axis=0)
        
        return all_points


    def save_image(self, idx, img_dir):
        
        # key, timestep = self._index_to_key_idx(idx)
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)
        
        try:
            if self.task == "stack_platforms2":
                img = self.f_original[key]["obs"]["world_camera_low_res_image"][timestep_start].astype(np.uint8)
            else:
                img = self.f_original[key]["obs"]["agentview_rgb"][timestep_start].astype(np.uint8)
        except Exception as e:
            print("skipping", idx, timestep_start, timestep_end, e)
            return None, None
        
        if self.flip_img:
            img = img[::-1]
        
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
    parser.add_argument('--img_key', type=str, default='primary', help='img key to use, one of [primary, secondary, tertiary]')    
    parser.add_argument('--target', type=str, default='path', help='path OR mask')
    parser.add_argument('--history', type=int, default=50, help='history length')
    parser.add_argument('--data_dir', type=str, default='/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/', help='data directory')
    parser.add_argument('--num_samples', type=int, default=np.inf, help='save num_samples')
    parser.add_argument('--num_seeds', type=int, default=1, help='number of seeds')
    parser.add_argument('--path_rdp_tolerance', type=float, default=0.05, help='tolerance for RDP path processing')
    parser.add_argument('--mask_rdp_tolerance', type=float, default=0.1, help='tolerance for RDP mask processing')
    parser.add_argument('--save_sketches_every_n', type=int, default=False, help='save every N-th sketch')
    parser.add_argument('--reword_quest', action="store_true", help='reword questions')
    parser.add_argument('--debug', action="store_true", help='debug mode')

    args = parser.parse_args()

    if args.task == "stack_platforms2":
        track_path = "/home/jeszhang/data/masked_vla_data/simpler_processed_data_subtrajectory_v2/dataset_movement_and_masks.h5"
        original_path = f"/lustre/fs12/portfolios/nvr/users/mmemmel/projects/simpler/data/stack_platforms2_roboverse.hdf5"
        
        tasks = [args.task]
        track_paths = [track_path]
        original_files = [original_path]
        
        # skip first to frames -> SimPLER specific
        start_idx = 2
        flip_img = False
    
    elif args.task == "libero_90":
        track_path = "/lustre/fs12/portfolios/nvr/users/mmemmel/jeszhang/masked_vla_data/libero_90_processed_256/dataset_movement_and_masks.h5"
        original_path = "/lustre/fs12/portfolios/nvr/users/mmemmel/projects/3dda/LIBERO/libero"
        original_path = os.path.join(original_path, args.task)

        original_files = [f for f in os.listdir(original_path) if f.endswith(".hdf5")]
        tasks = [f.replace(".hdf5", "") for f in original_files]
        original_files = [os.path.join(original_path, f) for f in original_files]
        track_paths = [track_path] * len(original_files)
        start_idx = 0
        flip_img = True
    elif "libero" in args.task:
        track_path = f"/lustre/fs12/portfolios/nvr/users/mmemmel/projects/3dda/LIBERO/libero/datasets/masked_vla_data/{args.task}_processed_256_05_12/dataset_movement_and_masks.h5"
        original_path = "/lustre/fs12/portfolios/nvr/users/mmemmel/projects/3dda/LIBERO/libero/datasets"
        original_path = os.path.join(original_path, args.task)

        original_files = [f for f in os.listdir(original_path) if f.endswith(".hdf5")]
        tasks = [f.replace(".hdf5", "") for f in original_files]
        original_files = [os.path.join(original_path, f) for f in original_files]
        track_paths = [track_path] * len(original_files)
        start_idx = 0
        flip_img = True

    else:
        raise NotImplementedError(f"Task {args.task} not implemented.")


    if args.debug:
        args.data_dir = os.path.join(args.data_dir, "debug")
        # if exists, delete args.data_dir
        if os.path.exists(args.data_dir):
            import shutil
            shutil.rmtree(args.data_dir)

        args.save_sketches_every_n = 1
    else:
        if args.task == "libero_90":
            args.data_dir = os.path.join(args.data_dir, f"sim_{args.target}_subtraj_hist")
        else:
            args.data_dir = os.path.join(args.data_dir, f"sim_{args.target}_subtraj_hist_ood")

    for task, track_path, original_path in zip(tasks, track_paths, original_files):

        # somehow didn't get tracked
        if "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket" in task:
            continue
        
        # copy track_path to /tmp/dataset_movement_and_masks.h5
        tmp_track_path = f"/tmp/{task}/dataset_movement_and_masks.h5"
        import shutil
        # if doesn't exist, copy
        # if not os.path.exists(tmp_track_path):
        # remove tmp_track_path if exists
        if os.path.exists(tmp_track_path):
            os.remove(tmp_track_path)
        os.makedirs(os.path.dirname(tmp_track_path), exist_ok=True)
        shutil.copy(track_path, tmp_track_path)
        track_path = tmp_track_path

        for seed in range(args.num_seeds):
            dataclass = TrackingDataclass(task, args.split, args.target, start_idx=start_idx, track_path=track_path, original_path=original_path, train_test_split=args.train_test_split, img_key=args.img_key, flip_img=flip_img, seed=seed, history=args.history)
            
            task_name = task + "_" + args.img_key + "_" + args.target + "_" + str(seed)
            convert_dataset(task=task_name, dataclass=dataclass, split=args.split, data_dir=args.data_dir, prompt_type=args.target, reword_quest=args.reword_quest, num_samples=args.num_samples, path_rdp_tolerance=args.path_rdp_tolerance, mask_rdp_tolerance=args.mask_rdp_tolerance, save_sketches_every_n=args.save_sketches_every_n)


# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_tracking.py --task stack_platforms2 --split train --target path --save_sketches_every_n 1000
# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_tracking.py --task stack_platforms2 --split test --target path --save_sketches_every_n 1000

# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_tracking.py --task stack_platforms2 --split train --target mask --save_sketches_every_n 1000
# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_tracking.py --task stack_platforms2 --split test --target mask --save_sketches_every_n 1000

# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_tracking.py --task stack_platforms2 --split train --target path_mask --save_sketches_every_n 1000
# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_tracking.py --task stack_platforms2 --split test --target path_mask --save_sketches_every_n 1000