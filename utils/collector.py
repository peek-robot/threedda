import os
import json
import h5py
import numpy as np

def norm_actions(actions, actions_min, actions_max):
    return 2 * (actions - actions_min) / (actions_max - actions_min) - 1

def denorm_actions(actions, actions_min, actions_max):
    return (actions + 1) * (actions_max - actions_min) / 2 + actions_min

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

        demo_grp.create_dataset("raw_actions", data=self.actions)

        self.f.flush()

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
            actions_norm = norm_actions(self.data_grp[dk]["raw_actions"][:], actions_min, actions_max)
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