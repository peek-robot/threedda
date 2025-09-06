# explicit masking baseline
server-client structure bc transformer versions are incompatible between SAM2 and 3DDA ...

## GroundedSAM2 server
```bash
conda activate pr_3dda_sam && python scripts/masking/server_groundedsam2.py
```

## 3DDA training w/ client inference
```bash
CUDA_VISIBLE_DEVICES=0 conda activate pr_3dda && python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_exp_mask.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_exp_mask --fps_subsampling_factor 5 --name ee_exp_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 0 --debug
```