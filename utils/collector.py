import os
import json
import h5py
import numpy as np

from utils.normalize import normalize

class DataCollector:
    def __init__(self, env, env_config, data_config, save_dir, out_file, train_valid_split, lang_key="lang_instr", clip_embedder=None):
        self.env = env
        self.env_config = env_config
        self.data_config = data_config
        self.save_dir = save_dir
        self.train_valid_split = train_valid_split
        self.lang_key = lang_key
        self.clip_embedder = clip_embedder

        self.demo_idx = -1
        os.makedirs(self.save_dir, exist_ok=True)
        self.f = h5py.File(os.path.join(self.save_dir, out_file), "w")
        self.data_grp = self.f.create_group("data")

    def reset(self):
        self.demo_idx += 1
        obs = self.env.reset()
        # reset dicts
        self.obs = {key: [obs[key]] for key in obs.keys()}
        self.actions = []

    def step(self, action):

        # exec absolute
        obs, r, done, info = self.env.step(action)
        self.actions.append(action)

        # update dicts
        for key in obs.keys():
            self.obs[key].append(obs[key])

    def save(self):
             
        demo_grp = self.data_grp.create_group(f"demo_{self.demo_idx}")

        obs_grp = demo_grp.create_group("obs")
        for key in self.obs.keys():
            if key == self.lang_key:
                self.obs[key] = np.repeat(self.clip_embedder.embed_instruction(self.obs[key][0]).numpy()[None], len(self.obs[key]), axis=0)
            obs_grp.create_dataset(key, data=self.obs[key][:-1])
        demo_grp.attrs.create("num_samples", len(self.obs[key][:-1]))

        demo_grp.create_dataset("actions_raw", data=self.actions)

        self.f.flush()

    def normalize_actions(self):
        actions = []
        for dk in self.data_grp.keys():
            actions.append(self.data_grp[dk]["actions_raw"][:])
        actions = np.concatenate(actions, axis=0)

        actions_min = np.min(actions, axis=0) # np.concatenate([self.env.min_qpos, [0.]]) # 
        actions_max = np.max(actions, axis=0) # np.concatenate([self.env.max_qpos, [1.]]) # 
        self.data_grp.attrs.create("actions_min", actions_min)
        self.data_grp.attrs.create("actions_max", actions_max)

        for dk in self.data_grp.keys():
            # normalize actions to [-1., 1.]
            actions_normalized = normalize(self.data_grp[dk]["actions_raw"][:], min=actions_min, max=actions_max)
            self.data_grp[dk].create_dataset("actions", data=actions_normalized, dtype=np.float32)

    def compute_path_and_mask(self):
        from utils.paths import generate_path_2d_from_obs, add_path_2d_to_img, add_mask_2d_to_img, smooth_path_rdp, scale_path
        for dk in self.data_grp.keys():

            H, W, C = self.data_grp[dk]["obs"]["rgb"][0].shape
            
            # compute SMOOTHED PATH in image space
            path_raw = generate_path_2d_from_obs(self.data_grp[dk]["obs"]["ee_pos"], self.data_grp[dk]["obs"]["camera_intrinsic"], self.data_grp[dk]["obs"]["camera_extrinsic"])
            path = scale_path(path_raw, min_in=0., max_in=H, min_out=0., max_out=1.)
            path = smooth_path_rdp(path, tolerance=0.05)
            path = scale_path(path, min_in=0., max_in=1., min_out=0., max_out=H).astype(np.int32)

            # store path as 2d image
            path_imgs = np.stack([add_path_2d_to_img(im, path) for im in self.data_grp[dk]["obs"]["rgb"]], axis=0)
            self.data_grp[dk]["obs"].create_dataset("path_rgb", data=path_imgs, dtype=np.uint8)
            self.obs["path_rgb"] = path_imgs

            # path shape (N, 2) pad N with zeros until N=P
            max_path_length = 255
            path_pad = np.pad(path, ((0, max_path_length - path.shape[0]), (0, 0)), mode="constant", constant_values=(-1.))
            # add batch dim (B, P, 2)
            path_pad = np.repeat(path_pad[None], path_imgs.shape[0], axis=0)
            # store raw path
            self.data_grp[dk]["obs"].create_dataset("path", data=path_pad, dtype=np.float32)
            
            # compute MASK in image space
            # obj 0 (& 1)
            obj_poses_raw = self.data_grp[dk]["obs"]["obj_poses"]
            obj_pose_0 = obj_poses_raw[:, :3]
            if obj_poses_raw.shape[-1] > 7:
                obj_pose_1 = obj_poses_raw[:, 7:10]
                obj_poses = np.concatenate([obj_pose_0, obj_pose_1], axis=0) # (N, 3)
            else:
                obj_poses = obj_pose_0 # (N, 3)

            cam_int = self.data_grp[dk]["obs"]["camera_intrinsic"]
            cam_ext = self.data_grp[dk]["obs"]["camera_extrinsic"]
            mask_raw = generate_path_2d_from_obs(obj_poses, np.concatenate((cam_int, cam_int)), np.concatenate((cam_ext, cam_ext)))
            mask = np.concatenate([path_raw, mask_raw], axis=0)

            # store mask as 2d depth
            mask_depths = np.stack([add_mask_2d_to_img(im, mask, mask_pixels=int(H * 0.15)) for im in self.data_grp[dk]["obs"]["depth"]], axis=0)
            self.data_grp[dk]["obs"].create_dataset("mask_depth", data=mask_depths, dtype=np.int16)
            self.obs["mask_depth"] = mask_depths

            # mask shape (N, 2) pad N with zeros until N=P
            max_mask_length = 255
            mask_pad = np.pad(mask, ((0, max_mask_length - mask.shape[0]), (0, 0)), mode="constant", constant_values=(-1.))
            # add batch dim (B, P, 2)
            mask_pad = np.repeat(mask_pad[None], mask_depths.shape[0], axis=0)
            # store raw mask
            self.data_grp[dk]["obs"].create_dataset("mask", data=mask_pad, dtype=np.float32)


    def close(self):
        self.compute_path_and_mask()
        self.normalize_actions()
        
        mask_grp = self.f.create_group("mask")
        
        data_keys = list(self.data_grp.keys())
        cutoff = int(len(data_keys) * self.train_valid_split)
        mask_grp.create_dataset("train", data=data_keys[:cutoff])
        mask_grp.create_dataset("valid", data=data_keys[cutoff:])

        self.data_grp.attrs.create("env_args", json.dumps({"env_name": str(type(self.env)), "type": str(type(self.env)), "env_kwargs": self.env_config, "data_kwargs": self.data_config}))

        self.f.flush()
        self.f.close()