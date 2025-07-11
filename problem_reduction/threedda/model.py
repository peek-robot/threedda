# import sys
# sys.path.append("/home/marius/Projects/3d_diffuser_actor")
# sys.path.append("/home/memmelma/Projects/robotic/3d_diffuser_actor")
import os
import torch
from torch import optim
import numpy as np

def get_model(da_config, device="cpu"):

    from diffuser_actor import DiffuserActor, DiffuserJointer

    assert da_config.action_space in ["abs_ee", "joint"], "Invalid action space: {}".format(da_config.action_space)

    if da_config.action_space == "abs_ee":
        model = DiffuserActor(
            backbone="clip",
            image_size=(256,256),
            embedding_dim=da_config.embedding_dim,
            num_vis_ins_attn_layers=2,
            use_instruction=True,
            fps_subsampling_factor=da_config.fps_subsampling_factor,
            # TODO: check those
            # gripper_loc_bounds=np.array([[-1.0, -1.0, 0], [1.0, 1.0, 1.0]]),
            gripper_loc_bounds=np.array([
                [0, -0.5, -0.01],
                [1, 0.5, 0.5]
            ]),
            # rotation_parametrization="quat", # -> 6D is hardcoded ...
            rotation_parametrization='6D',
            quaternion_format="xyzw",
            diffusion_timesteps=da_config.diffusion_timesteps,
            nhist=da_config.history,
            relative=False,
            lang_enhanced=False,
            loss_weights=[30., 10., 1.]
            # loss_weights=[30., 10., 0.]
        )
    elif da_config.action_space == "joint":
        model = DiffuserJointer(
            backbone="clip",
            image_size=da_config.image_size,
            embedding_dim=da_config.embedding_dim,
            num_attn_heads=da_config.num_attn_heads,
            num_vis_ins_attn_layers=2,
            use_instruction=True,
            fps_subsampling_factor=da_config.fps_subsampling_factor,
            high_res_features=da_config.high_res_features,
            gripper_loc_bounds=da_config.gripper_loc_bounds,
            joint_loc_bounds=da_config.joint_loc_bounds,
            loss_weights=da_config.loss_weights,
            diffusion_timesteps=da_config.diffusion_timesteps,
            nhist=da_config.history,
            relative=False,
            traj_relative=da_config.traj_relative,
            lang_enhanced=False
        )

    model.to(device)
    return model

def get_optimizer(model, lr=1e-4):
    """Initialize optimizer."""
    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0, "lr": lr},
        {"params": [], "weight_decay": 5e-4, "lr": lr}
    ]
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)
        else:
            optimizer_grouped_parameters[1]["params"].append(param)
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    return optimizer

def load_checkpoint(checkpoint, device="cpu"):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(checkpoint))

    ckpt_dict = torch.load(checkpoint, map_location="cpu")

    model_config = ckpt_dict["model_config"]
    model = get_model(model_config)
    model.load_state_dict(ckpt_dict["model_state_dict"])
    model.to(device)
    
    optimizer = get_optimizer(model, lr=model_config.lr)
    if 'optimizer_state_dict' in ckpt_dict:
        optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
        for p in range(len(optimizer.param_groups)):
            optimizer.param_groups[p]['lr'] = model_config.lr

    start_iter = ckpt_dict.get("iter", 0)
    best_loss = ckpt_dict.get("best_loss", None)

    wandb_dict = {
        "run_id": ckpt_dict.get("wandb_run_id", None),
        "entity": ckpt_dict.get("wandb_entity", None),
        "project": ckpt_dict.get("wandb_project", None),
    }

    print("=> loaded successfully '{}' (step {})".format(
        checkpoint, ckpt_dict.get("iter", 0)
    ))
    del ckpt_dict
    torch.cuda.empty_cache()
    return model, optimizer, start_iter, best_loss, wandb_dict, model_config

def save_checkpoint(
        model, optimizer,
        iter, new_loss, best_loss,
        wandb_config, model_config,
        log_dir
    ):
    """Save checkpoint if requested."""
    
    new_best = False
    if new_loss is None or best_loss is None or new_loss <= best_loss:
        best_loss = new_loss
        new_best = True

    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        
        "iter": iter + 1,
        "best_loss": best_loss,
        
        "wandb_run_id": wandb_config["run_id"],
        "wandb_entity": wandb_config["entity"],
        "wandb_project": wandb_config["project"],

        "model_config": model_config
    }

    if new_best:
        torch.save(save_dict, os.path.join(log_dir, "best.pth"))
    if (iter + 1) % 50 == 0:
        torch.save(save_dict, os.path.join(log_dir, f"iter_{iter+1}.pth"))
    torch.save(save_dict, os.path.join(log_dir, "last.pth"))
    return best_loss