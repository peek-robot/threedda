# data gen

```bash
conda activate pr_curobo_fix && python scripts/datagen/generate.py --task pick_and_place --num_samples 2500 --num_objs 3 --visual_augmentation --drop_failures --identifier ee
conda activate vila && CUDA_VISIBLE_DEVICES=1 python scripts/datagen/annotate_batch.py --model_path memmelma/vila_3b_blocks_path_mask_fast --path data/pick_and_place_2500_3_objs_va_ee.hdf5 --split_size 32
```

# model training

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_naive.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 3000 --task pick_and_place --name pnp_naive_fix_path --high_res_features --fps_subsampling_factor 25 --obs_continuous_gripper --obs_path --obs_mask_w_path --server_ip_vlm http://0.0.0.0:8000 --history 1 --update_every_timesteps_vlm 16 --resume --model_name_vlm vila_3b_path_mask_fast --mask_pixels=10 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_naive.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 3000 --task pick_and_place --name pnp_naive_fix_path_traj_relative --high_res_features --fps_subsampling_factor 25 --obs_continuous_gripper --obs_path --obs_mask_w_path --server_ip_vlm http://0.0.0.0:8000 --history 1 --update_every_timesteps_vlm 16 --resume --model_name_vlm vila_3b_path_mask_fast --traj_relative --mask_pixels=10 --resume
```