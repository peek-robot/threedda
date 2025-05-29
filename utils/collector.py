import os
import json
import h5py
import numpy as np
class DataCollector:
    def __init__(self, env, env_config, data_config, save_dir, out_file, train_valid_split, obs_keys=["rgb", "qpos", "obj_poses"]):
        self.env = env
        self.env_config = env_config
        self.data_config = data_config
        self.save_dir = save_dir
        self.train_valid_split = train_valid_split

        self.obs_keys = obs_keys
        self._reset_dicts()

        self.demo_idx = -1
        os.makedirs(self.save_dir, exist_ok=True)
        self.f = h5py.File(os.path.join(self.save_dir, out_file), "w")
        self.data_grp = self.f.create_group("data")

    def _reset_dicts(self):
        self.obs = {key: [] for key in self.obs_keys}
        self.actions = []

    def get_obs(self):
        for key in self.obs_keys:
            self.obs[key].append(getattr(self.env, f"get_{key}")())
    
    def reset(self):
        self.demo_idx += 1
        self.env.reset()
        self.get_obs()

    def step(self, action):

        self.actions.append(action)
        self.env.step(action)
        
        self.get_obs()

    def save(self):
             
        demo_grp = self.data_grp.create_group(f"demo_{self.demo_idx}")

        obs_grp = demo_grp.create_group("obs")
        for key in self.obs_keys:
            obs_grp.create_dataset(key, data=self.obs[key][:-1])
        demo_grp.attrs.create("num_samples", len(self.obs[key][:-1]))

        demo_grp.create_dataset("raw_actions", data=self.actions)

        self.f.flush()
        self._reset_dicts()

    def normalize_actions(self):
        actions = []
        for dk in self.data_grp.keys():
            actions.append(self.data_grp[dk]["raw_actions"][:])
        actions = np.concatenate(actions, axis=0)

        actions_min = np.min(actions, axis=0)
        actions_max = np.max(actions, axis=0)
        self.data_grp.attrs.create("actions_min", actions_min)
        self.data_grp.attrs.create("actions_max", actions_max)

        for dk in self.data_grp.keys():
            # normalize actions to [-1., 1.]
            actions_norm = 2 * (self.data_grp[dk]["raw_actions"][:] - actions_min) / (actions_max - actions_min) - 1
            self.data_grp[dk].create_dataset("actions", data=actions_norm, dtype=np.float32)

    def close(self):
        self.normalize_actions()

        mask_grp = self.f.create_group("mask")
        
        data_keys = list(self.data_grp.keys())
        cutoff = int(len(data_keys) * self.train_valid_split)
        mask_grp.create_dataset("train", data=data_keys[:cutoff])
        mask_grp.create_dataset("valid", data=data_keys[cutoff:])

        self.data_grp.attrs.create("env_args", json.dumps({"env_name": str(type(self.env)), "type": str(type(self.env)), "env_kwargs": self.env_config, "data_kwargs": self.data_config}))

        self.f.flush()
        self.f.close()