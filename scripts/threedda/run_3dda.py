# pip uninstall transformers -y
# pip install transformers --no-cache-dir

# https://github.com/easydiffusion/easydiffusion/issues/1851#issuecomment-2437615074
# pip install huggingface_hub==0.25.0

import argparse
import json
import os
import h5py
import cv2
import einops

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import trange
import torchvision

import wandb
from torch.utils.data import DataLoader

from problem_reduction.threedda.criterion import TrajectoryCriterion
from problem_reduction.threedda.model import (
    get_model,
    get_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from problem_reduction.threedda.data import prepare_batch
from problem_reduction.threedda.vis_utils import plot_actions_and_log_wandb

import json
import random

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory

from torch.utils.data import DataLoader

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_dataloaders_from_mimic(config):

    ObsUtils.initialize_obs_utils_with_config(config)

    action_keys = ["actions"]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        action_keys=action_keys,
        language_conditioned=config.observation.language_conditioned,
        verbose=True,
    )

    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    train_sampler = trainset.get_dataset_sampler()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        persistent_workers=config.train.num_data_workers > 0,
        drop_last=True,
        pin_memory=True,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    return train_loader, valid_loader


def train(
    task,
    model,
    optimizer,
    train_loader,
    test_loader,
    best_loss=None,
    start_epoch=0,
    device=None,
    wandb_config=None,
    dataset=None,
    model_config=None,
    output_dir=None,
    server_ip_vlm=None,
    model_name_vlm=None,
    update_every_timesteps_vlm=32,
):

    criterion = TrajectoryCriterion()
    train_loader_iter = iter(train_loader)

    if test_loader is not None:
        test_loader_iter = iter(test_loader)

    for epoch in range(start_epoch, model_config.num_epochs + 1):
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

            batch_prepared = prepare_batch(
                batch,
                history=model_config.history,
                horizon=model_config.horizon,
                obs_noise_std=model_config.obs_noise_std,
                obs_path_mask_noise_std=model_config.obs_path_mask_noise_std,
                obs_no_proprio=model_config.obs_no_proprio,
                obs_discrete_gripper=not model_config.obs_continuous_gripper,
                obs_crop=model_config.obs_crop,
                obs_crop_cube=model_config.obs_crop_cube,
                obs_outlier=model_config.obs_outlier,
                obs_hamster=model_config.obs_hamster,
                obs_path=model_config.obs_path,
                obs_mask=model_config.obs_mask,
                obs_mask_w_path=model_config.obs_mask_w_path,
                mask_pixels=model_config.mask_pixels,
                obs_gt=model_config.obs_gt,
                rainbow_path=model_config.rainbow_path,
                action_space=model_config.action_space,
                device=device,
            )

            # forward and backward
            loss_dict = model.forward(**batch_prepared, run_inference=False)
            loss_dict["total_loss"].backward()
            optimizer.step()
            
            for k, v in loss_dict.items():
                if k not in train_losses:
                    train_losses[k] = []
                train_losses[k].append(v.item())

        # compute metrics
        if epoch % 10 == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():

                acts = model.forward(**batch_prepared, run_inference=True)

                if model_config.action_space == "joint":
                    metrics = criterion.compute_metrics_joint(
                        acts,
                        batch_prepared["gt_trajectory"],
                        batch_prepared["trajectory_mask"],
                    )[0]
                else:
                    metrics = criterion.compute_metrics(
                        acts,
                        batch_prepared["gt_trajectory"],
                        batch_prepared["trajectory_mask"],
                    )[0]

                for k, v in metrics.items():
                    if k not in train_losses:
                        train_losses[k] = []
                    train_losses[k].append(v.item())

        # logging
        wandb.log(
            {
                "epoch": epoch,
                **{f"train/{k}": sum(v) / len(v) for k, v in train_losses.items()},
            }
        )

        # plot actions
        if epoch > 500 and epoch % model_config.eval_every_n_epochs == 0  and epoch > 0:
            act_dim = acts.shape[-1]
            plot_actions_and_log_wandb(
                acts.cpu().numpy().reshape(-1, act_dim),
                batch_prepared["gt_trajectory"].cpu().numpy().reshape(-1, act_dim),
                "train/actions",
                epoch,
            )

        # plot observations
        if epoch == 0:

            # log clip features
            rgb_prep = batch_prepared["rgb_obs"][:9]
            rgb_prep = einops.rearrange(rgb_prep, "bt ncam c h w -> (bt ncam) c h w")
            rgb_prep = model.encoder.normalize(rgb_prep)
            with torch.no_grad():
                x = model.encoder.backbone(rgb_prep)["res3"]
            feature_map = x.cpu().mean(1)
            feature_map_resized = np.stack(
                [cv2.resize(i, rgb_prep.shape[-2:]) for i in feature_map.numpy()]
            )
            import matplotlib.cm as cm

            feature_map_colored = (
                cm.inferno(feature_map_resized)[..., :3].transpose(0, 3, 1, 2) * 255.0
            )

            feature_map = torch.from_numpy(feature_map_colored)  # .unsqueeze(1)
            feature_grid = torchvision.utils.make_grid(
                feature_map, nrow=3, normalize=True, padding=2
            )
            wandb.log(
                {"epoch": epoch, "train/obs_rgb_feature": wandb.Image(feature_grid)}
            )

            # log rgb obs
            rgb_grid = torchvision.utils.make_grid(
                batch_prepared["rgb_obs"][:9, 0], nrow=3, normalize=True, padding=2
            )
            wandb.log({"epoch": epoch, "train/obs_rgb": wandb.Image(rgb_grid)})

            # log pcd obs
            z = batch_prepared["pcd_obs"][:, :, 2, :, :]
            z_grid = torchvision.utils.make_grid(
                z[:9], nrow=3, normalize=True, padding=2
            )
            wandb.log({"epoch": epoch, "train/obs_pcd": wandb.Image(z_grid)})

        # save checkpoint
        best_loss = save_checkpoint(
            model,
            optimizer,
            epoch,
            sum(train_losses["total_loss"]) / len(train_losses["total_loss"]),
            best_loss,
            wandb_config,
            model_config,
            output_dir,
        )

        # compute eval metrics
        if False: #epoch % model_config.eval_every_n_epochs == 0 and epoch > 0:
        # if epoch > 300 and model_config.eval_every_n_epochs == 0 and epoch > 0:
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

                batch_prepared = prepare_batch(
                    batch,
                    history=model_config.history,
                    horizon=model_config.horizon,
                    obs_noise_std=0.0, # model_config.obs_noise_std,
                    obs_path_mask_noise_std=0.0, # model_config.obs_path_mask_noise_std,
                    obs_discrete_gripper=not model_config.obs_continuous_gripper,
                    obs_no_proprio=model_config.obs_no_proprio,
                    obs_crop=model_config.obs_crop,
                    obs_crop_cube=model_config.obs_crop_cube,
                    obs_outlier=model_config.obs_outlier,
                    obs_hamster=model_config.obs_hamster,
                    obs_path=model_config.obs_path,
                    obs_mask=model_config.obs_mask,
                    obs_mask_w_path=model_config.obs_mask_w_path,
                    mask_pixels=model_config.mask_pixels,
                    obs_gt=model_config.obs_gt,
                    rainbow_path=model_config.rainbow_path,
                    action_space=model_config.action_space,
                    device=device,
                )

                with torch.no_grad():
                    loss_dict = model.forward(**batch_prepared, run_inference=False)
                    for k, v in loss_dict.items():
                        if k not in test_losses:
                            test_losses[k] = []
                        test_losses[k].append(v.item())

                    acts = model.forward(**batch_prepared, run_inference=True)
                    if model_config.action_space == "joint":
                        metrics = criterion.compute_metrics_joint(
                            acts,
                            batch_prepared["gt_trajectory"],
                            batch_prepared["trajectory_mask"],
                        )[0]
                    else:
                        metrics = criterion.compute_metrics(
                            acts,
                            batch_prepared["gt_trajectory"],
                            batch_prepared["trajectory_mask"],
                        )[0]

                    for k, v in metrics.items():
                        if k not in test_losses:
                            test_losses[k] = []
                        test_losses[k].append(v.item())
            # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"test/{k}": sum(v) / len(v) for k, v in test_losses.items()},
                }
            )
            act_dim = acts.shape[-1]
            plot_actions_and_log_wandb(
                acts.cpu().numpy().reshape(-1, act_dim),
                batch_prepared["gt_trajectory"].cpu().numpy().reshape(-1, act_dim),
                "test/actions",
                epoch,
            )

            rgb_grid = torchvision.utils.make_grid(
                batch_prepared["rgb_obs"][:9, 0], nrow=3, normalize=True, padding=2
            )
            wandb.log({"epoch": epoch, "test/obs_rgb": wandb.Image(rgb_grid)})

            z = batch_prepared["pcd_obs"][:, :, 2, :, :]
            z_grid = torchvision.utils.make_grid(
                z[:9], nrow=3, normalize=True, padding=2
            )
            wandb.log({"epoch": epoch, "test/obs_pcd": wandb.Image(z_grid)})

        # compute eval success
        if epoch % model_config.eval_every_n_epochs == 0 and epoch != 0:
        # if epoch > 500 and epoch % model_config.eval_every_n_epochs == 0 and epoch != 0:
            
            eval_mode = "closed_loop"
            
            if task == "pick_and_place" or task =="pick":
                if task == "pick_and_place":
                    n_rollouts = 25
                    n_steps = 128
                elif task == "pick":
                    n_rollouts = 16
                    n_steps = 72

                from run_3dda_eval import eval_3dda
                successes, videos, instructions = eval_3dda(
                    task=task,
                    policy=model,
                    model_config=model_config,
                    data_path=dataset,
                    mode=eval_mode,
                    action_chunking=True,
                    action_chunk_size=8,
                    n_rollouts=n_rollouts,
                    n_steps=n_steps,
                    obs_path=model_config.obs_path,
                    obs_mask=model_config.obs_mask,
                    obs_mask_w_path=model_config.obs_mask_w_path,
                    obs_hamster=model_config.obs_hamster,
                    mask_pixels=model_config.mask_pixels,
                    rainbow_path=model_config.rainbow_path,
                    obs_gt=model_config.obs_gt,
                    obs_exp_mask=model_config.obs_exp_mask,
                    server_ip_vlm=server_ip_vlm,
                    model_name_vlm=model_name_vlm,
                    update_every_timesteps_vlm=update_every_timesteps_vlm,
                    seed=model_config.seed,
                )
            elif task == "pick_and_place_robo":
                if task == "pick_and_place_robo":
                    n_rollouts = 5
                    n_steps = 192
                from run_3dda_eval_robo import eval_3dda
                successes, videos, instructions = eval_3dda(
                    task=task,
                    policy=model,
                    model_config=model_config,
                    data_path=dataset,
                    mode=eval_mode,
                    action_chunking=True,
                    action_chunk_size=8,
                    n_rollouts=n_rollouts,
                    n_steps=n_steps,
                    obs_path=model_config.obs_path,
                    obs_mask=model_config.obs_mask,
                    obs_mask_w_path=model_config.obs_mask_w_path,
                    obs_hamster=model_config.obs_hamster,
                    mask_pixels=model_config.mask_pixels,
                    rainbow_path=model_config.rainbow_path,
                    obs_gt=model_config.obs_gt,
                    obs_exp_mask=model_config.obs_exp_mask,
                    server_ip_vlm=server_ip_vlm,
                    model_name_vlm=model_name_vlm,
                    update_every_timesteps_vlm=update_every_timesteps_vlm,
                    seed=model_config.seed,
                )

            wandb.log(
                {"epoch": epoch, f"eval/{eval_mode}/success_rate": np.mean(successes)}
            )
            for i, (video, instruction) in enumerate(zip(videos, instructions)):
                wandb.log(
                    {
                        "epoch": epoch,
                        f"eval/{eval_mode}/{instruction}_{i}": wandb.Video(
                            video.transpose(0, 3, 1, 2), fps=10, format="mp4"
                        ),
                    }
                )

        # ### HACK: real data eval ###
        # if (epoch % model_config.eval_every_n_epochs == 0 and epoch != 0):
        #     eval_mode = "open_loop"
        #     successes, videos, instructions = eval_3dda(
        #         policy=model,
        #         model_config=model_config,
        #         data_path="gifs_curobo/pick_10_1_objs_va_high_cam.hdf5",

        #         real_data_path="pick_10_1_objs_va_high_cam_real.hdf5",
        #         open_loop_obs_key="obs_real",
        #         n_rollouts=1,
        #         n_steps=1,

        #         mode=eval_mode,
        #         action_chunking=True,
        #         action_chunk_size=8,
        #         clip_embedder=clip_embedder,
        #         path_mode="open_loop" if model_config.obs_path else None,
        #         mask_mode="open_loop" if model_config.obs_mask else None,
        #     )
        #     wandb.log({"epoch": epoch, f"eval_real/{eval_mode}/success_rate": np.mean(successes)})
        #     for i, (video, instruction) in enumerate(zip(videos, instructions)):
        #         wandb.log(
        #             {
        #                 "epoch": epoch,
        #                 f"eval_real/{eval_mode}/{instruction}_{i}": wandb.Video(
        #                     video.transpose(0, 3, 1, 2), fps=10, format="gif"
        #                 )
        #             }
        #         )
        # ### HACK: real data eval ###


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser(
        description="robomimic training script for SimPLER (RoboVerse)"
    )
    parser.add_argument("--name", type=str, required=True, help="experiment name")
    parser.add_argument("--task", type=str, required=True, help="task")
    parser.add_argument(
        "--action_space", type=str, default="joint", help="action space"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=500,
        help="number of epochs",
    )
    parser.add_argument(
        "--eval_every_n_epochs",
        type=int,
        default=100,
        help="evaluate every n epochs",
    )
    parser.add_argument(
        "--epoch_every_n_steps",
        type=int,
        default=100,
        help="train every n steps",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/red_cube_2500_pcd_vanilla.hdf5",
        help="path to dataset",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results/",
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
        "--image_size",
        type=int,
        default=128,
        help="image size",
    )
    parser.add_argument(
        "--obs_crop",
        action="store_true",
        help="crop points",
    )
    parser.add_argument(
        "--obs_crop_cube",
        action="store_true",
        help="crop cube points",
    )
    parser.add_argument(
        "--obs_outlier",
        action="store_true",
        help="remove outliers",
    )
    parser.add_argument(
        "--obs_path",
        action="store_true",
        help="use path rgb",
    )
    parser.add_argument(
        "--obs_mask",
        action="store_true",
        help="use mask",
    )
    parser.add_argument(
        "--obs_mask_w_path",
        action="store_true",
        help="use mask w/ path",
    )
    parser.add_argument(
        "--obs_gt",
        action="store_true",
        help="use gt",
    )
    parser.add_argument(
        "--obs_hamster",
        action="store_true",
        help="use hamster",
    )
    parser.add_argument(
        "--obs_exp_mask",
        action="store_true",
        help="use explicit mask",
    )
    parser.add_argument(
        "--obs_noise_std",
        type=float,
        default=0.01,
        help="noise std",
    )
    parser.add_argument(
        "--obs_path_mask_noise_std",
        type=float,
        default=0.0,
        help="noise std",
    )
    parser.add_argument(
        "--fps_subsampling_factor",
        type=int,
        default=5,
        help="fps subsampling factor",
    )
    parser.add_argument(
        # NOTE: increases initial points from 1024 -> 4096 by increasing CLIP feature resolution from 32x32 to 64x64
        "--high_res_features",
        action="store_true",
        help="use high res features",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=2,
        help="history",
    )
    parser.add_argument(
        "--augment_pcd",
        action="store_true",
        help="augment pcd",
    )
    parser.add_argument(
        "--augment_rgb",
        action="store_true",
        help="augment rgb",
    )
    parser.add_argument(
        "--normalize_loss",
        action="store_true",
        help="normalize loss",
    )
    parser.add_argument(
        "--obs_continuous_gripper",
        action="store_true",
        help="continuous gripper",
    )
    parser.add_argument(
        "--obs_no_proprio",
        action="store_true",
        help="no proprio",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="slurm",
    )
    parser.add_argument(
        "--traj_relative",
        action="store_true",
        help="traj relative",
    )
    parser.add_argument(
        "--model_name_vlm",
        type=str,
        default="vila_3b_path_mask_fast",
        help="model name vlm",
    )
    parser.add_argument(
        "--server_ip_vlm",
        type=str,
        default=None,
        help="server ip vlm",
    )
    parser.add_argument(
        "--update_every_timesteps_vlm",
        type=int,
        default=32,
        help="update every timesteps vlm",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug",
    )
    parser.add_argument(
        "--mask_pixels",
        type=int,
        default=10,
        help="mask pixels",
    )
    parser.add_argument(
        "--rainbow_path",
        action="store_true",
        help="rainbow path",
    )
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="+",
        default=[30., 10., 1.],
        help="loss weights",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed",
    )
    # parse arguments
    args = parser.parse_args()

    if args.debug:
        args.name = "debug"
        args.resume = False
        args.epoch_every_n_steps = 3
        args.eval_every_n_epochs = 1

    args.name = f"{args.name}_{args.seed}"
    seed_everything(args.seed)

    print(json.dumps(vars(args)))

    # fix CUDA issue with DataLoader
    try:
        mp.set_start_method("spawn")
    except RuntimeError as e:
        print(e)

    output_dir = os.path.join(args.outdir, args.name)
    os.makedirs(output_dir, exist_ok=True)

    joint_loc_bounds = np.array(
        [
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        ]
    )

    from argparse import Namespace
    model_config = {
        "num_epochs": args.num_epochs,
        "eval_every_n_epochs": args.eval_every_n_epochs,
        "epoch_every_n_steps": args.epoch_every_n_steps,
        "horizon": args.horizon,
        "history": args.history,
        "batch_size": 64, # 32,
        "lr": 3e-4,
        "embedding_dim": 60, # -> maybe try 32 | 60
        "num_attn_heads": 6, # -> maybe try 4 | 6
        "diffusion_timesteps": 100,
        "action_space": args.action_space,  # "joint" or "abs_ee"
        "traj_relative": args.traj_relative,
        "joint_loc_bounds": joint_loc_bounds,
        "gripper_loc_bounds": None,
        "loss_weights": args.loss_weights, # [30, 10]
        "normalize_loss": not args.normalize_loss,
        "image_size": (args.image_size, args.image_size),
        # ablations
        "fps_subsampling_factor": args.fps_subsampling_factor, # -> maybe sample less
        "obs_noise_std": args.obs_noise_std,
        "obs_path_mask_noise_std": args.obs_path_mask_noise_std,
        "obs_no_proprio": args.obs_no_proprio,
        "obs_continuous_gripper": args.obs_continuous_gripper,
        "obs_crop": args.obs_crop,
        "obs_crop_cube": args.obs_crop_cube,
        "obs_outlier": args.obs_outlier,
        "obs_path": args.obs_path,
        "obs_mask": args.obs_mask,
        "obs_mask_w_path": args.obs_mask_w_path,
        "obs_exp_mask": args.obs_exp_mask,
        "rainbow_path": args.rainbow_path,
        "mask_pixels": args.mask_pixels,
        "obs_gt": args.obs_gt,
        "obs_hamster": args.obs_hamster,
        "augment_pcd": args.augment_pcd,
        "augment_rgb": args.augment_rgb,
        "high_res_features": args.high_res_features,
        "seed": args.seed,
    }
    model_config = Namespace(**model_config)

    low_dim_keys = [
        "ee_pose" if args.action_space == "abs_ee" else "qpos",
        # "qpos_normalized",
        "gripper_state_continuous",
        "gripper_state_discrete",
        "lang_instr",
    ]

    # set path/mask keys depending on whether to use gt or vlm generated predictions
    if model_config.obs_path or model_config.obs_hamster:
        low_dim_keys.append("path" if model_config.obs_gt else "path_vlm")
    if model_config.obs_mask:
        low_dim_keys.append("mask" if model_config.obs_gt else "mask_vlm")
    if model_config.obs_mask_w_path:
        low_dim_keys.append("mask" if model_config.obs_gt else "mask_vlm")
        low_dim_keys.append("path" if model_config.obs_gt else "path_vlm")

    if args.slurm:
        import shutil

        tmp_dataset = "/tmp/" + os.path.basename(args.dataset)
        if not os.path.exists(tmp_dataset):
            print("Copying dataset to /tmp/ ...")
            shutil.copy(args.dataset, tmp_dataset)
        else:
            print("dataset already exists in /tmp/")
        args.dataset = tmp_dataset

    validate = False # True
    print(("WARNING: NOT VALIDATING"))
    ext_cfg = {
        "algo_name": "bc",
        "experiment": {
            "validate": validate,
        },
        "observation": {
            "modalities": {
                "obs": {
                    "low_dim": low_dim_keys,
                    "rgb": ["rgb"],
                    "depth": [],
                    "scan": [],
                    "pc": (["depth", "camera_intrinsic", "camera_extrinsic"]),
                },
            }
        },
        "train": {
            "data": args.dataset,
            "output_dir": "../bc_transformer_trained_models",
            "num_data_workers": 4,
            "hdf5_cache_mode": None,
            "hdf5_use_swmr": True,
            "hdf5_load_next_obs": False,
            "hdf5_normalize_obs": False,
            "hdf5_filter_key": "train" if validate else None,
            "hdf5_validation_filter_key": "valid" if validate else None,
            "seq_length": model_config.horizon + 1,
            "pad_seq_length": True,
            "frame_stack": model_config.history,
            "pad_frame_stack": True,
            "dataset_keys": [],
            "goal_mode": None,
            "cuda": True,
            "batch_size": model_config.batch_size,
            "num_epochs": model_config.num_epochs,
            "seed": model_config.seed,
        },
    }
    mimic_config = config_factory("bc")
    with mimic_config.values_unlocked():
        mimic_config.update(ext_cfg)

    train_loader, test_loader = get_dataloaders_from_mimic(mimic_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume_path = os.path.join(output_dir, "last.pth")
    if args.resume and os.path.exists(resume_path):

        model, optimizer, start_epoch, best_loss, wandb_config, model_config = (
            load_checkpoint(resume_path, device=device)
        )
        # model_config.num_epochs = 2500

        wandb.init(
            entity=wandb_config["entity"],
            project=wandb_config["project"],
            resume="must",
            id=wandb_config["run_id"],
        )

    else:
        start_epoch = 0
        best_loss = None

        model = get_model(model_config, device=device)

        optimizer = get_optimizer(model, lr=model_config.lr)

        configs = {
            "verse_config": mimic_config,
            "da_config": vars(model_config),
        }
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=configs)
        wandb.run.name = f"{args.name}_{wandb.run.name}"

        wandb_config = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "run_id": wandb.run.id,
        }

    # HACK to resume training with new num_epochs
    model_config.num_epochs = args.num_epochs

    train(
        task=args.task,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        best_loss=best_loss,
        start_epoch=start_epoch,
        device=device,
        wandb_config=wandb_config,
        dataset=args.dataset,
        model_config=model_config,
        output_dir=output_dir,
        model_name_vlm=args.model_name_vlm,
        server_ip_vlm=args.server_ip_vlm,
        update_every_timesteps_vlm=args.update_every_timesteps_vlm,
    )
