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
        assert target in ['path', 'mask', 'path_mask', 'path_mask_lang', 'path_mask_history', 'path_mask_history_lang'], f"Target {target} not found."

        self.task = task
        self.movement_key = "movement_across_subtrajectory" if "processed" in self.task else "movement_across_video"

        self.split = split
        self.target = target

        self.f_track = h5py.File(track_path, 'r', swmr=True)[task]

        self.img_key_to_name = OXE_DATASET_CONFIGS[task]["image_obs_keys"]
        self.f_original = tfds.load(task, data_dir=tfds_path, split="train")
        self.ds_iter = iter(self.f_original)

        self.img_key = img_key

        self.curr_sample = None
        self.curr_timestep = None
        self.curr_idx = None

        # seed train/test split
        self.seed = seed
        np.random.seed(self.seed)

        # HACK: fix initial seed for bridge_v2 to avoid test set leakage
        if task == "bridge_v2":
            np.random.seed(0)
        
        train_keys = np.random.choice(list(self.f_track.keys()), int(len(self.f_track)*train_test_split), replace=False)
        test_keys = [k for k in self.f_track.keys() if k not in train_keys]
        
        self.demo_keys = train_keys if split == "train" else test_keys
        # HACK: sort demo_key [0, 1, 2, ...] for compatibility w/ step_dataset
        self.demo_keys = sorted(self.demo_keys, key=lambda x: int(x.split('_')[1]))
        self.key_tag = self.demo_keys[0].split("_")[0] + "_" # e.g., "episode_"
        self.quest = None
        self.path = None
        self.path_history = None

    def get_indices(self):
        sub_idcs = []
        for dk in self.demo_keys:
            if self.img_key in self.f_track[dk].keys():
                n_sub_idcs = len(self.f_track[dk][self.img_key]["traj_splits_indices"])
                # for all sub-trajectories in demo key
                for sub_idx in np.arange(0, n_sub_idcs-1):
                    sub_idcs.append(f"{dk.replace(self.key_tag, '')}_{sub_idx}")
        return sub_idcs


    def _load_data(self, idx):
        
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)
        
        for i, step in enumerate(iter(self.curr_sample["steps"])):
            if i == timestep_start:
                self.img = step["observation"][self.img_key_to_name[self.img_key]].numpy().astype(np.uint8)
                try:
                    self.quest = step["observation"]["natural_language_instruction"].numpy().decode("utf-8")
                except:
                    self.quest = step["language_instruction"].numpy().decode("utf-8")
                break
        
        if "path" in self.target:
            path = self.f_track[key][self.img_key]["gripper_positions"][timestep_start:timestep_end]
            self.path = path

        if "history" in self.target:
            # 2) get history of fixed length
            history = np.random.randint(20, 60)
            path_history = self.f_track[key][self.img_key]["gripper_positions"][max(0, timestep_start-history):timestep_start]
            self.path_history = path_history

        if "mask" in self.target:
            significant_points = self.f_track[key][self.img_key]["significant_points"]
            # movement_across_video = self.f_track[key]["primary"][self.movement_key]
            stopped_points = self.f_track[key][self.img_key]["stopped_points"]
            current_gripper_position = self.f_track[key][self.img_key]["gripper_positions"][timestep_start]
            all_points = np.concatenate([significant_points[timestep_start].astype(np.uint16), stopped_points[timestep_start].astype(np.uint16)])
            all_points = np.unique(all_points, axis=0)
            self.mask = all_points

        if "lang" in self.target:
            try:
                self.lang = self.f_track[key][self.img_key]["trajectory_labels"][lang_idx].decode('utf-8')
            except:
                self.lang = ""
            print("QUEST", self.quest, "LANG", self.lang)

    def step(self, idx):
        
        idx_str = str(idx)
        idx = [int(i) for i in idx.split("_")]

        # if first, step once
        if self.curr_idx is None:
            self.curr_sample = next(self.ds_iter)
            self.curr_idx = 0
        # if idx is greater than curr_idx, step until idx is reached
        elif idx[0] > self.curr_idx:
            while idx[0] > self.curr_idx:
                self.curr_sample = next(self.ds_iter)
                self.curr_idx += 1
        # just for safety
        elif idx[0] < self.curr_idx:
            raise ValueError(f"Cannot go backwards to index {idx[0]} from {self.curr_idx}")

        # load new data
        self._load_data(idx_str)

    # def _idx_to_sub_idx(self, idx):
    #     idx = [int(i) for i in idx.split("_")]
    #     key = f"{self.key_tag}{idx[0]}"
    #     traj_splits_indices = self.f_track[key][self.img_key]["traj_splits_indices"]
    #     timestep_start = traj_splits_indices[idx[1]]
    #     timestep_end = traj_splits_indices[idx[1]+1]

    #     # 1) sample point in sub-trajectory
    #     timestep_start = timestep_start + np.random.randint(0, timestep_end - timestep_start)

    #     lang_idx = idx[1]

    #     # HACK: skip first 20% of first sub-traj
    #     if timestep_start == 0:
    #         traj_len = timestep_end - timestep_start
    #         timestep_start = timestep_start + int(traj_len*0.2)

    #     return key, timestep_start, timestep_end, lang_idx

    def _idx_to_sub_idx(self, idx):
        idx = [int(i) for i in idx.split("_")]
        key = f"{self.key_tag}{idx[0]}"
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

    def load_lang(self, idx):
        return self.lang

    def load_quest(self, idx):
        return self.quest
        
    def load_path(self, idx):
        return self.path

    def load_path_history(self, idx):
        return self.path_history

    def load_mask(self, idx):
        return self.mask

    def save_image(self, idx, img_dir):
        
        key, timestep_start, timestep_end, lang_idx = self._idx_to_sub_idx(idx)

        # # if img is empty, skip
        # if img is None or img.sum() == 0:
        #     print("skipping", idx, timestep_start, timestep_end)
        #     return None, None

        # # resize to make sure img is scaled corresponding to path/mask
        # if img.shape[:2] != mask_shape:
        #     print("resizing", img.shape[:2], "to", mask_shape)
        #     img = cv2.resize(img, mask_shape)

        img_name = self.split + "_" + self.task + "_" + str(idx) + ".jpg"
        img_path = os.path.join(img_dir, img_name)

        os.makedirs(img_dir, exist_ok=True)
        Image.fromarray(self.img).save(img_path)
        
        return self.img, img_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None, help='custom task')
    parser.add_argument('--split', type=str, default='train', help='train or test split')
    parser.add_argument('--train_test_split', type=float, default=1.0, help='train (train_test_split) test (1-train_test_split) split')
    parser.add_argument('--img_key', type=str, default='primary', help='img key to use, one of [primary, secondary, tertiary]')
    parser.add_argument('--target', type=str, default='path', help='path OR mask')
    parser.add_argument('--data_dir', type=str, default='/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/', help='data directory')
    parser.add_argument('--num_samples', type=int, default=np.inf, help='save num_samples')
    # parser.add_argument('--num_seeds', type=int, default=1, help='number of seeds')
    parser.add_argument('--seed', type=int, default=0, help='seed')
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
        args.data_dir = os.path.join(args.data_dir, f"oxe_processed_{args.target}_subtraj_hist")
        args.data_dir = args.data_dir + "_rw" if args.reword_quest else args.data_dir

    import sys
    sys.path.append("/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/")
    from oxe_configs import OXE_DATASET_CONFIGS, DATASET_TRANSFORMS

    # prevent TFDS from taking up all GPU memory
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow_datasets as tfds
    
    # tfds_path = "/lustre/fs12/portfolios/nvr/projects/nvr_srl_simpler/datasets/open_x_embodiment/"
    if args.task == "droid" or args.task == "bridge_v2":
        tfds_path = "/lustre/fs12/portfolios/nvr/projects/nvr_srl_simpler/datasets/open_x_embodiment_rlds/"
    else:
        tfds_path = "/lustre/fs12/portfolios/nvr/projects/nvr_srl_simpler/datasets/open_x_embodiment_rlds/done/"
        
    dataset_names = [x.split()[0] for x in DATASET_TRANSFORMS]
    if args.task is not None:
        dataset_names = [args.task]

    pbar = tqdm(total=len(dataset_names))
    for i, dataset_name in enumerate(dataset_names):
        
        track_path = "/lustre/fs12/portfolios/nvr/users/mmemmel/jeszhang/masked_vla_data/oxe_processed_subtrajectory/"
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

        dataclass = TrackingDataclass(dataset_name, args.split, args.target, track_path=track_path, tfds_path=tfds_path, train_test_split=args.train_test_split, img_key=args.img_key, seed=args.seed)

        task_name = dataset_name + "_" + args.img_key + "_" + args.target + "_" + str(args.seed)
        task_name = task_name + "_rw" if args.reword_quest else task_name

        convert_dataset(task=task_name, dataclass=dataclass, split=args.split, data_dir=args.data_dir, prompt_type=args.target, reword_quest=args.reword_quest, num_samples=args.num_samples, path_rdp_tolerance=args.path_rdp_tolerance, mask_rdp_tolerance=args.mask_rdp_tolerance, save_sketches_every_n=args.save_sketches_every_n)

        # for i in dataclass.get_indices()[:args.num_samples]:

            
        #     dataclass.step(i)

        #     img, img_path = dataclass.save_image(i, ".")
        #     path = dataclass.load_path(i)

        #     # if path is None or img is None:
        #     #     continue

        #     from vila_utils.utils.decode import add_path_2d_to_img

        #     sketch = add_path_2d_to_img(img, path, cmap="inferno")
        #     Image.fromarray(sketch).save(f"train_{dataset_name}_{i}_sketch.png")