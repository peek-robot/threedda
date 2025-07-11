import os
import argparse
import numpy as np
from PIL import Image

from torch import Generator
from torch.utils.data import random_split

import roboverse.constants as c
from roboverse.configs import get_cfg_defaults
from roboverse.unifiers import PATH2D_Unifier

from vila_utils.utils.convert import convert_dataset

class RoboVerseDataclass():
    
    def __init__(self, task, split, train_test_split=0.9, seed=0):
        
        simpler_tasks = [
            "stack_platforms2",
            "procedural_cabinet_left",
            "procedural_cabinet_right"]

        libero_tasks = [
            "libero_10",
            "libero_90"]

        assert task in simpler_tasks or task in libero_tasks, f"Task {task} not found in mapping."
        assert split in ['train', 'test'], f"Split {split} not found."

        self.task = task
        self.split = split

        roboverse_config = get_cfg_defaults()

        roboverse_config.unifier = c.PATH2D
        roboverse_config.horizon = 1
        roboverse_config.history = -1
        roboverse_config.has_path = False
        roboverse_config.has_depth = True

        if self.task in simpler_tasks:
            roboverse_config.datasets = [c.SIMPLER]
            roboverse_config.SIMPLER.tasks = [self.task]
            roboverse_config.SIMPLER.cam_list = [c.THREE_P1]
            roboverse_config.SIMPLER.has_depth = False

        elif self.task in libero_tasks:

            roboverse_config.datasets = [c.LIBERO]
            roboverse_config.LIBERO.task_suites = [self.task]
            roboverse_config.LIBERO.cam_list = [c.THREE_P1]
            roboverse_config.LIBERO.base_path = "/lustre/fs12/portfolios/nvr/users/mmemmel/projects/3dda/LIBERO/libero/datasets"
            roboverse_config.LIBERO.has_depth = False
            roboverse_config.LIBERO.has_path = True

        path_dataset = PATH2D_Unifier(roboverse_config)

        train_size = int(train_test_split * len(path_dataset))
        test_size = len(path_dataset) - train_size

        gen = Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(path_dataset, [train_size, test_size], generator=gen)
        self.dataset = train_dataset if self.split == 'train' else test_dataset
    
    def get_indices(self):
        return range(len(self.dataset))

    def load_quest(self, idx):
        return self.dataset[idx]["instr"]

    def load_path(self, idx):
        return  self.dataset[idx]["path_2d"][0]
    
    def save_image(self, idx, img_dir):
        img = self.dataset[idx]["rgb"][0,0]
        
        img_name = self.split + "_" + self.task + "_" + str(idx) + ".jpg"
        img_path = os.path.join(img_dir, img_name)

        os.makedirs(img_dir, exist_ok=True)
        Image.fromarray(img).save(img_path)
        
        return img, img_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='task to convert')
    parser.add_argument('--split', type=str, default='train', help='train or test split')
    parser.add_argument('--data_dir', type=str, default='/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/', help='data directory')
    parser.add_argument('--num_samples', type=int, default=np.inf, help='save num_samples')
    parser.add_argument('--save_sketches_every_n', type=int, default=False, help='save every N-th sketch')
    parser.add_argument('--reword_quest', action="store_true", help='reword questions')
    parser.add_argument('--debug', action="store_true", help='debug mode')

    args = parser.parse_args()

    task = args.task

    dataclass = RoboVerseDataclass(task, args.split)

    if args.debug:
        args.data_dir = os.path.join(args.data_dir, "debug")

    convert_dataset(task=task, dataclass=dataclass, split=args.split, data_dir=args.data_dir, prompt_type="path", reword_quest=args.reword_quest, num_samples=args.num_samples, save_sketches_every_n=args.save_sketches_every_n)\

# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_roboverse.py --task libero_90 --split train --save_sketches_every_n 1000
# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_roboverse.py --task libero_90 --split test --save_sketches_every_n 1000
# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_roboverse.py --task libero_10 --split train --save_sketches_every_n 1000
# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_roboverse.py --task libero_10 --split test --save_sketches_every_n 1000

# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_roboverse.py --task stack_platforms2 --split train --save_sketches_every_n 1000
# python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_roboverse.py --task stack_platforms2 --split test --save_sketches_every_n 1000