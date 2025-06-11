import os
import json
import h5py
import numpy as np

from utils.normalize import normalize

class DataCollector:
    def __init__(self, env, env_config, data_config, save_dir, out_file, train_valid_split):
        self.env = env
        self.env_config = env_config
        self.data_config = data_config
        self.save_dir = save_dir
        self.train_valid_split = train_valid_split

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
        obs, r, done, info = self.env.step(action)
        # update dicts
        self.actions.append(action)
        for key in obs.keys():
            self.obs[key].append(obs[key])

    def save(self):
             
        demo_grp = self.data_grp.create_group(f"demo_{self.demo_idx}")

        obs_grp = demo_grp.create_group("obs")
        for key in self.obs.keys():
            obs_grp.create_dataset(key, data=self.obs[key][:-1])
        demo_grp.attrs.create("num_samples", len(self.obs[key][:-1]))

        demo_grp.create_dataset("actions_raw", data=self.actions)

        self.f.flush()

    def normalize_actions(self):
        actions = []
        for dk in self.data_grp.keys():
            actions.append(self.data_grp[dk]["actions_raw"][:])
        actions = np.concatenate(actions, axis=0)

        actions_min = np.concatenate([self.env.min_qpos, [0.]]) #np.min(actions, axis=0)
        actions_max = np.concatenate([self.env.max_qpos, [1.]]) # np.max(actions, axis=0)
        self.data_grp.attrs.create("actions_min", actions_min)
        self.data_grp.attrs.create("actions_max", actions_max)

        for dk in self.data_grp.keys():
            # normalize actions to [-1., 1.]
            actions_normalized = normalize(self.data_grp[dk]["actions_raw"][:], min=actions_min, max=actions_max)
            self.data_grp[dk].create_dataset("actions", data=actions_normalized, dtype=np.float32)
    
    def compute_paths(self):
        from utils.paths import generate_path_2d_from_obs, add_path_2d_to_img, smooth_path_rdp, scale_path
        for dk in self.data_grp.keys():
            path = generate_path_2d_from_obs(self.data_grp[dk]["obs"])
            # TODO dynamically set resolution
            path = scale_path(path, min_in=0., max_in=224., min_out=0., max_out=1.)
            path = smooth_path_rdp(path, tolerance=0.05)
            path = scale_path(path, min_in=0., max_in=1., min_out=0., max_out=224.).astype(np.int32)

            # store path as 2d image
            path_imgs = np.stack([add_path_2d_to_img(im, path) for im in self.data_grp[dk]["obs"]["rgb"]], axis=0)
            self.data_grp[dk]["obs"].create_dataset("path_img", data=path_imgs, dtype=np.uint8)

            # path shape (N, 2) pad N with zeros until N=P
            path_pad = np.pad(path, ((0, 127 - path.shape[1]), (0, 0)), mode="constant", constant_values=(0))
            # add batch dim (B, P, 2)
            path_pad = np.repeat(path_pad[None], self.data_grp[dk].attrs["num_samples"], axis=0)
            # store raw path
            self.data_grp[dk]["obs"].create_dataset("path", data=path_pad, dtype=np.float32)
            
            # import matplotlib.pyplot as plt
            # plt.imsave(f"path_img_{dk}.png", path_imgs[0])
            # import IPython; IPython.embed()

    def close(self):
        self.compute_paths()
        self.normalize_actions()

        mask_grp = self.f.create_group("mask")
        
        data_keys = list(self.data_grp.keys())
        cutoff = int(len(data_keys) * self.train_valid_split)
        mask_grp.create_dataset("train", data=data_keys[:cutoff])
        mask_grp.create_dataset("valid", data=data_keys[cutoff:])

        self.data_grp.attrs.create("env_args", json.dumps({"env_name": str(type(self.env)), "type": str(type(self.env)), "env_kwargs": self.env_config, "data_kwargs": self.data_config}))

        self.f.flush()
        self.f.close()