import json
import h5py
import torch
import imageio
import os
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from PIL import Image

from problem_reduction.threedda.text_embed import CLIPTextEmbedder
from problem_reduction.threedda.data import prepare_batch
from problem_reduction.threedda.model import load_checkpoint
from problem_reduction.threedda.vis_utils import plot_actions

from problem_reduction.robot.wrappers.framestack import FrameStackWrapper
from problem_reduction.robot.robot_env import CubeEnv
from problem_reduction.utils.normalize import denormalize

from problem_reduction.vila.inference_helpers import vila_inference_api

def add_vlm_predictions(obs, instructions, timestep, update_every_timesteps=15, model_name="vila_3b_blocks_path_mask_fast", server_ip=None, obs_path=False, obs_mask=False, vlm_cache=None, vlm_cache_step=0):
    if vlm_cache is None or vlm_cache_step < np.floor(timestep / update_every_timesteps).astype(int):
        vlm_cache_step = np.floor(timestep / update_every_timesteps).astype(int)
        print("Querying VLM at timestep", timestep, "...")
        
        image, path_pred, mask_pred = vila_inference_api(obs["rgb"], instructions[-1], model_name=model_name, server_ip=server_ip, prompt_type="path_mask")
    
        vlm_cache = {
            "path_pred": path_pred,
            "mask_pred": mask_pred,
        }
    
    if obs_path:
        obs["path_vlm"] = vlm_cache["path_pred"]
    if obs_mask:
        obs["mask_vlm"] = vlm_cache["mask_pred"]

    return obs, vlm_cache, vlm_cache_step

def eval_3dda(
    data_path,

    policy=None,
    model_config=None,
    ckpt_path=None,

    mode="closed_loop",
    action_chunking=True,
    action_chunk_size=8,

    n_rollouts=10,
    n_steps=112,

    obs_path=False,
    obs_mask=False,
    obs_mask_w_path=False,
    mask_pixels=10,
    model_name_vlm="peek_3b",
    server_ip_vlm=None,
    update_every_timesteps_vlm=32,
    seed=1,
):
    
    # load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt_path is not None and (policy is None or model_config is None):
        policy, _, _, _, wandb_dict, model_config = load_checkpoint(ckpt_path)
        policy = policy.to(device)

    clip_embedder = CLIPTextEmbedder()
    clip_embedder = clip_embedder.to(device)

    # load env + normalization data
    with h5py.File(data_path, "r", swmr=True) as f:
        env_config = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
        action_min, action_max = (
            f["data"].attrs["actions_min"],
            f["data"].attrs["actions_max"],
        )

    # set to ood seed if not using path or mask
    combined_seed = env_config["seed"] + 1 + seed
    
    # init framestack
    num_frames = model_config.history
    framestack = FrameStackWrapper(num_frames=num_frames)

    successes = []
    videos = []
    instructions = []
    for i in trange(n_rollouts, desc="ROLLOUT"):

        env = CubeEnv(**env_config)
        env.seed(combined_seed + i)

        obs = env.reset()

        obs["lang_instr"] = clip_embedder.embed_instruction(obs["lang_instr"])

        instructions.append(env.get_lang_instr())

        imgs = [np.concatenate([obs["rgb"], env.get_obs()["rgb"]], axis=1)]

        if obs_path or obs_mask or obs_mask_w_path:
            # initial vlm predictions and cache
            obs, vlm_cache, vlm_cache_step = add_vlm_predictions(obs, instructions, timestep=0, update_every_timesteps=update_every_timesteps_vlm, model_name=model_name_vlm, server_ip=server_ip_vlm, obs_path=obs_path, obs_mask=obs_mask or obs_mask_w_path)
        
        framestack.reset()
        framestack.add_obs(obs)

        pred_actions = []
        assert n_steps % action_chunk_size == 0
        for j in range(n_steps // action_chunk_size):

            obs = framestack.get_obs_history()

            act_queue = []

            # add batch dimension, convert to torch
            sample = {
                "obs": {
                    k: torch.from_numpy(v[None]).to(device) for k, v in obs.items()
                }
            }
            # preprocess same as training
            batch_prepared = prepare_batch(
                sample,
                history=model_config.history,
                horizon=model_config.horizon,
                obs_noise_std=0.0,
                obs_path_mask_noise_std=0.0,
                obs_discrete_gripper=not model_config.obs_continuous_gripper,
                obs_path=obs_path,
                obs_mask=obs_mask,
                obs_mask_w_path=obs_mask_w_path,
                mask_pixels=mask_pixels,
                device=device,
                action_space="abs_ee",
            )

            if obs_path:
                rgb_obs = (
                    batch_prepared["rgb_obs"][0, 0].permute(1, 2, 0).cpu().numpy()
                    * 255.0
                )
                imgs.append(rgb_obs.astype(np.uint8))

            # [B, T, 7+1]
            with torch.no_grad():
                acts = policy.forward(**batch_prepared, run_inference=True)

            if action_chunking:
                chunk_size = action_chunk_size
            else:
                chunk_size = 1
            for i, act in enumerate(acts[0].cpu().numpy()[:chunk_size]):
                # discretize gripper action to ensure gripper_state is (0., 1.) as during data gen
                act[7] = 1.0 if act[7] > 0.5 else 0.0
                act_queue.append(act)

            for t, act in zip(reversed(range(len(act_queue))), act_queue):
                pred_actions.append(act)
                
                obs, r, done, info = env.step(act)

                obs["lang_instr"] = clip_embedder.embed_instruction(obs["lang_instr"])

                if obs_path or obs_mask or obs_mask_w_path:
                    # initial vlm predictions and cache
                    timestep=j*action_chunk_size
                    update_every_timesteps=update_every_timesteps_vlm
                    if not "rgb" in obs.keys() and vlm_cache_step < np.floor(timestep / update_every_timesteps).astype(int):
                        obs["rgb"] = env.get_obs()["rgb"]
                    # update vlm predictions and cache -> only compute new path/mask predictions every 15 steps
                    obs, vlm_cache, vlm_cache_step = add_vlm_predictions(obs, instructions, timestep=j*action_chunk_size, update_every_timesteps=update_every_timesteps_vlm, model_name=model_name_vlm, server_ip=server_ip_vlm, obs_path=obs_path, obs_mask=obs_mask or obs_mask_w_path, vlm_cache=vlm_cache, vlm_cache_step=vlm_cache_step)
                
                framestack.add_obs(obs)
                if "rgb" in obs.keys():
                    imgs.append(obs["rgb"])
                else:
                    imgs.append(env.get_obs()["rgb"])

                success = env.is_success(task="pick_and_place")
                if success:
                    break
            if success:
                break

        successes.append(success)
        try:
            videos.append(np.array(imgs))
        except:
            videos.append(np.array([im[:128, :128, :3] for im in imgs]))

    print(
        f"{mode} {'act chunking' if action_chunking else ''} Success rate: {np.mean(successes)}"
    )
    return successes, videos, instructions

