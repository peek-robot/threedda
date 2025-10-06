import os
import torch
from torch import optim
import numpy as np

def small_random_rotation_matrix_batch(B, max_angle=3 * torch.pi / 180, device='cpu'):
    angles = torch.empty(B, 3, device=device).uniform_(-max_angle, max_angle)
    ax, ay, az = angles[:, 0], angles[:, 1], angles[:, 2]

    cx, cy, cz = torch.cos(angles.T)
    sx, sy, sz = torch.sin(angles.T)

    zeros = torch.zeros(B, device=device)
    ones = torch.ones(B, device=device)

    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, cx, -sx], dim=1),
        torch.stack([zeros, sx, cx], dim=1)
    ], dim=1)

    Ry = torch.stack([
        torch.stack([cy, zeros, sy], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-sy, zeros, cy], dim=1)
    ], dim=1)

    Rz = torch.stack([
        torch.stack([cz, -sz, zeros], dim=1),
        torch.stack([sz, cz, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)

    return torch.bmm(Rz, torch.bmm(Ry, Rx))

def augment_pointcloud_batch(pc, translation_range=0.03, rotation_range=3, noise_range=0.01):
    # pc: [B, T, 3, H, W]
    B, T, C, H, W = pc.shape
    assert C == 3, "Expected 3 channels for XYZ data"
    device = pc.device

    pc = pc.permute(0, 1, 3, 4, 2).contiguous()  # [B, T, H, W, 3]
    pc_flat = pc.view(B, T, H * W, 3)  # [B, T, N, 3]

    # Translation: [B, 1, 1, 3] (same for all points in B)
    translation = torch.empty(B, 1, 1, 3, device=device).uniform_(-translation_range, translation_range)
    pc_flat = pc_flat + translation

    # Rotation: [B, 3, 3]
    R = small_random_rotation_matrix_batch(B, max_angle=rotation_range * torch.pi / 180, device=device)
    pc_flat = torch.matmul(pc_flat, R.transpose(1, 2).unsqueeze(1))  # [B, T, N, 3]

    # Gaussian noise: [B, T, N, 3] (per point)
    noise = torch.randn(B, T, H * W, 3, device=device) * noise_range
    pc_flat = pc_flat + noise

    pc = pc_flat.view(B, T, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 3, H, W]
    return pc

def augment_rgb_sequence(imgs, brightness=0.2, contrast=0.2, color_jitter=0.1):
    # imgs: [B, T, C, H, W]
    device = imgs.device
    B, T, C, H, W = imgs.shape

    # Brightness: [B, 1, 1, 1]
    brightness_shift = (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * brightness
    imgs = imgs + brightness_shift.unsqueeze(1)  # [B, T, C, H, W]

    # Contrast: [B, 1, 1, 1]
    contrast_scale = 1 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * contrast
    mean = imgs.mean(dim=(3, 4), keepdim=True)  # [B, T, C, 1, 1]
    imgs = (imgs - mean) * contrast_scale.unsqueeze(1) + mean

    # Color jitter: [B, C, 1, 1]
    jitter = 1 + (torch.rand(B, C, 1, 1, device=device) * 2 - 1) * color_jitter
    imgs = imgs * jitter.unsqueeze(1)  # [B, T, C, H, W]

    return imgs.clamp(0, 1)

def get_model(da_config, device="cpu"):

    from diffuser_actor import DiffuserActor

    model = DiffuserActor(
        backbone="clip",
        image_size=da_config.image_size,
        embedding_dim=da_config.embedding_dim,
        num_attn_heads=da_config.num_attn_heads,
        num_vis_ins_attn_layers=2,
        use_instruction=True,
        fps_subsampling_factor=da_config.fps_subsampling_factor,
        gripper_loc_bounds=np.array([
            [0.25, -0.25, 0.0],
            [0.75, 0.25, 0.4]
        ]),
        rotation_parametrization='6D',
        quaternion_format="wxyz",
        diffusion_timesteps=da_config.diffusion_timesteps,
        nhist=da_config.history,
        relative=False,
        lang_enhanced=False,
        loss_weights=da_config.loss_weights,
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