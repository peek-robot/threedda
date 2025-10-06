
# mask
CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_mask --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 0 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_mask --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 1 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_mask --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 2 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_mask --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 3 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_mask --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 4 --resume

# path 

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 0 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 1 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 2 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 3 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 4 --resume

# path + mask
CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --obs_mask_w_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 0 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --obs_mask_w_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 1 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --obs_mask_w_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 2 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --obs_mask_w_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 3 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --obs_mask_w_path --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_path_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 4 --resume

# lang
CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_lang_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 0 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_lang_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 1 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_lang_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 2 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_lang_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 3 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://0.0.0.0:8000 --name ee_lang_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --mask_pixels 15 --seed 4 --resume




<!-- CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_hamster.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm hamster_13b --server_ip_vlm https://edclduajln.a.pinggy.link --name ee_hamster_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_hamster --obs_path_mask_noise_std 0.01 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm https://edclduajln.a.pinggy.link --name ee_path_rainbow_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path --rainbow_path --obs_path_mask_noise_std 0.01 --resume -->

# fulltraj path 

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_fulltraj.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm abl_vila_3b_path_fulltraj --server_ip_vlm http://0.0.0.0:8000 --name ee_full_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 0 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_fulltraj.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm abl_vila_3b_path_fulltraj --server_ip_vlm http://0.0.0.0:8000 --name ee_full_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 1 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_fulltraj.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm abl_vila_3b_path_fulltraj --server_ip_vlm http://0.0.0.0:8000 --name ee_full_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 2 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_fulltraj.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm abl_vila_3b_path_fulltraj --server_ip_vlm http://0.0.0.0:8000 --name ee_full_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 3 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_fulltraj.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_path --fps_subsampling_factor 5 --model_name_vlm abl_vila_3b_path_fulltraj --server_ip_vlm http://0.0.0.0:8000 --name ee_full_path_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 4 --resume


# HAMSTER

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_hamster.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm hamster_13b --server_ip_vlm http://edclduajln.a.pinggy.link --name ee_hamster_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --obs_hamster --seed 0 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_hamster.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm hamster_13b --server_ip_vlm http://edclduajln.a.pinggy.link --name ee_hamster_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --obs_hamster --seed 1 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_hamster.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm hamster_13b --server_ip_vlm http://edclduajln.a.pinggy.link --name ee_hamster_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --obs_hamster --seed 2 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_hamster.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm hamster_13b --server_ip_vlm http://edclduajln.a.pinggy.link --name ee_hamster_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --obs_hamster --seed 3 --resume

CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_hamster.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --fps_subsampling_factor 5 --model_name_vlm hamster_13b --server_ip_vlm http://edclduajln.a.pinggy.link --name ee_hamster_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --obs_hamster --seed 4 --resume

# explicit masking

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_exp_mask.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_exp_mask --fps_subsampling_factor 5 --name ee_exp_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 0 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_exp_mask.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_exp_mask --fps_subsampling_factor 5 --name ee_exp_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 1 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_exp_mask.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_exp_mask --fps_subsampling_factor 5 --name ee_exp_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 2 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_exp_mask.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_exp_mask --fps_subsampling_factor 5 --name ee_exp_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 3 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset data/pick_and_place_2500_3_objs_va_vel_004_ee_exp_mask.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place --obs_exp_mask --fps_subsampling_factor 5 --name ee_exp_mask_abl --history 1 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 1000 --seed 4 --resume

# robosuite

conda activate pr_3dda_robocasa && CUDA_VISIBLE_DEVICES=1 python scripts/threedda/run_3dda.py --dataset /home/memmelma/Projects/tool_use/data/PnP/trajs_joint_s2r_1_02.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place_robo --obs_path --obs_mask_w_path --fps_subsampling_factor 5 --history 1 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://edclduajln.a.pinggy.link --mask_pixels 15 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 32 --seed 0 --resume --name ee_robo_1_2k

conda activate pr_3dda_robocasa && CUDA_VISIBLE_DEVICES=0 python scripts/threedda/run_3dda.py --dataset /home/memmelma/Projects/tool_use/data/PnP/trajs_joint_s2r_1_02.hdf5 --augment_pcd --augment_rgb --obs_noise_std 0.01 --num_epochs 1500 --task pick_and_place_robo --obs_path --obs_mask_w_path --fps_subsampling_factor 5 --history 2 --model_name_vlm vila_3b_path_mask_fast --server_ip_vlm http://edclduajln.a.pinggy.link --mask_pixels 15 --obs_continuous_gripper --action_space abs_ee --obs_path_mask_noise_std 0.01 --update_every_timesteps_vlm 32 --seed 0 --resume --name ee_robo_1_2k_hist_2