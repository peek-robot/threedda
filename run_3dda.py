# pip uninstall transformers -y
# pip install transformers --no-cache-dir

# https://github.com/easydiffusion/easydiffusion/issues/1851#issuecomment-2437615074
# pip install huggingface_hub==0.25.0

import argparse
import json
import os

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from tqdm import trange
import torchvision

import wandb
from torch.utils.data import DataLoader

import sys
sys.path.append("/home/memmelma/Projects/robotic")
from threedda.criterion import TrajectoryCriterion
from threedda.utils import get_model, get_optimizer, load_checkpoint, save_checkpoint, plot_actions_and_log_wandb, prepare_batch
    
import json

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.config import config_factory

from torch.utils.data import DataLoader

def get_dataloaders_from_mimic(config):

    ObsUtils.initialize_obs_utils_with_config(config)

    action_keys = ["actions"]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=config.train.data,
            all_obs_keys=config.all_obs_keys,
            action_keys = action_keys,
            language_conditioned=config.observation.language_conditioned,
            verbose=True)

    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()

    # initialize data loaders
    train_loader = DataLoader(dataset=trainset,
                                sampler=train_sampler,
                                batch_size=config.train.batch_size,
                                shuffle=(train_sampler is None),
                                num_workers=config.train.num_data_workers,
                                drop_last=True)

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(dataset=validset,
                                  sampler=valid_sampler,
                                  batch_size=config.train.batch_size,
                                  shuffle=(valid_sampler is None),
                                  num_workers=num_workers,
                                  drop_last=True)
    else:
        valid_loader = None

    return train_loader, valid_loader

def train(
    model,
    optimizer,
    clip_embedder,

    train_loader,
    test_loader,

    best_loss=None,
    start_epoch=0,

    device=None,
    wandb_config=None,
    model_config=None,
    output_dir=None,
):

    criterion = TrajectoryCriterion()
    train_loader_iter = iter(train_loader)
    test_loader_iter = iter(test_loader)

    for epoch in range(start_epoch, model_config.num_epochs):
        train_losses = {}

        for _ in trange(
            model_config.epoch_every_n_steps,
            desc=f"TRAIN {epoch} / {model_config.num_epochs}",
        ):

            
            model.train()
            optimizer.zero_grad()
            
            # get next batch
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                # reset for next dataset pass
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)

            batch_prepared = prepare_batch(batch, clip_embedder, history=model_config.history, horizon=model_config.horizon, obs_noise_std=model_config.obs_noise_std, obs_crop=model_config.obs_crop, obs_path=model_config.obs_path, device=device)
            
            # forward and backward
            loss_dict = model.forward(**batch_prepared, run_inference=False)
            loss_dict["total_loss"].backward()
            optimizer.step()
            for k, v in loss_dict.items():
                if k not in train_losses:
                    train_losses[k] = []
                train_losses[k].append(v.item())
        
        # compute metrics
        model.eval()
        with torch.no_grad():

            acts = model.forward(**batch_prepared, run_inference=True)
            if model_config.action_space == "joint":
                metrics = criterion.compute_metrics_joint(acts, batch_prepared["gt_trajectory"], batch_prepared["trajectory_mask"])[0]
            else:
                metrics = criterion.compute_metrics(acts, batch_prepared["gt_trajectory"], batch_prepared["trajectory_mask"])[0]
            
            for k, v in metrics.items():
                if k not in train_losses:
                    train_losses[k] = []
                train_losses[k].append(v.item())

        # logging
        wandb.log({"epoch": epoch, **{f"train/{k}": sum(v) / len(v) for k, v in train_losses.items()}})
        act_dim = acts.shape[-1]
        plot_actions_and_log_wandb(acts.cpu().numpy().reshape(-1, act_dim), batch_prepared["gt_trajectory"].cpu().numpy().reshape(-1, act_dim), "train/actions", epoch)

        rgb_grid = torchvision.utils.make_grid(batch_prepared["rgb_obs"][:9,0], nrow=3,  normalize=True, padding=2)
        wandb.log({"epoch": epoch, "train/obs_rgb": wandb.Image(rgb_grid)})

        z = batch_prepared["pcd_obs"][:,:,2,:,:]
        z_grid = torchvision.utils.make_grid(z[:9], nrow=3,  normalize=True, padding=2)
        wandb.log({"epoch": epoch, "train/obs_pcd": wandb.Image(z_grid)})

        # save checkpoint
        best_loss = save_checkpoint(
            model, optimizer,
            epoch, sum(train_losses["total_loss"]) / len(train_losses["total_loss"]), best_loss,
            wandb_config, model_config,
            output_dir
        )

        if epoch % 25 == 0 and epoch > 0:
            test_losses = {
                "total_loss": [],
            }
            model.eval()

            for _ in trange(
                len(test_loader),
                desc=f"TEST {epoch} / {model_config.num_epochs}",
            ):
                
                try:
                    batch = next(test_loader_iter)
                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    batch = next(test_loader_iter)

                batch_prepared = prepare_batch(batch, clip_embedder, history=model_config.history, horizon=model_config.horizon, obs_noise_std=model_config.obs_noise_std, obs_crop=model_config.obs_crop, obs_path=model_config.obs_path, device=device)

                with torch.no_grad():
                    loss_dict = model.forward(**batch_prepared, run_inference=False)
                    for k, v in loss_dict.items():
                        if k not in test_losses:
                            test_losses[k] = []
                        test_losses[k].append(v.item())

                    acts = model.forward(**batch_prepared, run_inference=True)
                    if model_config.action_space == "joint":
                        metrics = criterion.compute_metrics_joint(acts, batch_prepared["gt_trajectory"], batch_prepared["trajectory_mask"])[0]
                    else:
                        metrics = criterion.compute_metrics(acts, batch_prepared["gt_trajectory"], batch_prepared["trajectory_mask"])[0]
                    
                    for k, v in metrics.items():
                        if k not in test_losses:
                            test_losses[k] = []
                        test_losses[k].append(v.item())
            # logging
            wandb.log({"epoch": epoch, **{f"test/{k}": sum(v) / len(v) for k, v in test_losses.items()}})
            act_dim = acts.shape[-1]
            plot_actions_and_log_wandb(acts.cpu().numpy().reshape(-1, act_dim), batch_prepared["gt_trajectory"].cpu().numpy().reshape(-1, act_dim), "test/actions", epoch)

            rgb_grid = torchvision.utils.make_grid(batch_prepared["rgb_obs"][:9,0], nrow=3,  normalize=True, padding=2)
            wandb.log({"epoch": epoch, "test/obs_rgb": wandb.Image(rgb_grid)})

            z = batch_prepared["pcd_obs"][:,:,2,:,:]
            z_grid = torchvision.utils.make_grid(z[:9], nrow=3,  normalize=True, padding=2)
            wandb.log({"epoch": epoch, "test/obs_pcd": wandb.Image(z_grid)})


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser(
        description="robomimic training script for SimPLER (RoboVerse)"
    )
    parser.add_argument("--name", type=str, required=True, help="experiment name")
    parser.add_argument("--action_space", type=str, default="joint", help="action space")

    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/memmelma/Projects/robotic/gifs_curobo/red_cube_2500_pcd_vanilla.hdf5",
        help="path to dataset",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/memmelma/Projects/robotic/results/",
        help="path to output directory",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sim2real",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="memmelma",
        help="wandb entity",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from latest ckpt"
    )

    # EXPERIMENTS

    parser.add_argument(
        "--obs_crop",
        action="store_true",
        help="crop points",
    )
    parser.add_argument(
        "--obs_path",
        action="store_true",
        help="use path rgb",
    )
    parser.add_argument(
        "--obs_noise_std",
        type=float,
        default=0.01,
        help="noise std",
    )
    parser.add_argument(
        "--fps_subsampling_factor",
        type=int,
        default=5,
        help="fps subsampling factor",
    )

    # parse arguments
    args = parser.parse_args()

    print(json.dumps(vars(args)))

    # fix CUDA issue with DataLoader
    mp.set_start_method("spawn")

    output_dir = os.path.join(args.outdir, args.name)
    os.makedirs(output_dir, exist_ok=True)

    from argparse import Namespace
    model_config = {
        "num_epochs": 500,
        "epoch_every_n_steps": 100,
        "horizon": 16,
        "history": 2,
        "batch_size": 128,
        "lr": 3e-4,
        "embedding_dim": 60,
        "num_attn_heads": 6,
        "diffusion_timesteps": 25, # -> CALVIN: 25, else: 100
        "action_space": args.action_space, # "joint" or "abs_ee"

        # ablations
        "fps_subsampling_factor": args.fps_subsampling_factor,
        "obs_noise_std": args.obs_noise_std,
        "obs_crop": args.obs_crop,
        "obs_path": args.obs_path,
    }
    model_config = Namespace(**model_config)

    ext_cfg = {
        "algo_name": "bc",
        "experiment": {
            "validate": True,
        },
        "observation": {
            "modalities": {
                "obs": {
                    # "low_dim": ["qpos", "gripper_state", "path"],
                    "low_dim": ["qpos", "gripper_state"],
                    "rgb": ["rgb"] if not model_config.obs_path else ["path_img"],
                    "depth": [],
                    "scan": [],
                    "pc": ["depth", "camera_intrinsic", "camera_extrinsic"]
                },
            }
        },
        "train": {
            "data": args.dataset,
            "output_dir": "../bc_transformer_trained_models",
            "num_data_workers": 0,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "hdf5_load_next_obs": False,
            "hdf5_normalize_obs": False,
            "hdf5_filter_key": "train",
            "hdf5_validation_filter_key": "valid",
            "seq_length": model_config.horizon + 1,
            "pad_seq_length": True,
            "frame_stack": model_config.history,
            "pad_frame_stack": True,
            "dataset_keys": [
                "actions"
            ],
            "goal_mode": None,
            "cuda": True,
            "batch_size": model_config.batch_size,
            "num_epochs": model_config.num_epochs,
            "seed": 1
        }
    }
    mimic_config = config_factory("bc")
    with mimic_config.values_unlocked():
        mimic_config.update(ext_cfg)
    
    train_loader, test_loader = get_dataloaders_from_mimic(mimic_config)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from threedda.text_embed import CLIPTextEmbedder
    clip_embedder = CLIPTextEmbedder()
    clip_embedder = clip_embedder.to(device)
    
    if args.resume:
        
        model, optimizer, start_epoch, best_loss, wandb_config, model_config = load_checkpoint(os.path.join(output_dir, "last.pth"))
        model.to(device)

        wandb.init(entity=wandb_config["entity"], project=wandb_config["project"], resume="must", id=wandb_config["run_id"])
    
    else:
        start_epoch = 0
        best_loss = None

        model = get_model(model_config)
        model = model.to(device)
        optimizer = get_optimizer(model, lr=model_config.lr)

        configs = {
            "verse_config": mimic_config,
            "da_config": model_config,
        }
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=configs)
        wandb.run.name = f"{args.name}_{wandb.run.name}"

        wandb_config = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "run_id": wandb.run.id,
        }

    train(
        model,
        optimizer,
        clip_embedder,

        train_loader,
        test_loader,

        best_loss=best_loss,
        start_epoch=start_epoch,

        device=device,
        wandb_config=wandb_config,
        model_config=model_config,
        output_dir=output_dir,
    )