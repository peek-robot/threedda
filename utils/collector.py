import os
import h5py

class DataCollector:
    def __init__(self, env, save_dir, obs_keys=["rgb", "qpos", "obj_pose"], actions_keys=["qpos", "gripper_pos"]):
        self.env = env
        self.save_dir = save_dir
        
        self.obs_keys = obs_keys
        self.actions_keys = actions_keys
        self._reset_dicts()

        self.demo_idx = -1
        os.makedirs(self.save_dir, exist_ok=True)
        self.f = h5py.File(os.path.join(self.save_dir, "mujoco.hdf5"), "w")

    def _reset_dicts(self):
        self.obs = {key: [] for key in self.obs_keys}
        self.actions = {key: [] for key in self.actions_keys}

    def get_obs(self):
        for key in self.obs_keys:
            self.obs[key].append(getattr(self.env, f"get_{key}")())
    
    def reset(self):
        self.demo_idx += 1
        self.env.reset()
        self.get_obs()

    def step(self, actions):

        for key in self.actions_keys:
            self.actions[key].append(actions[key])
        
        self.env.step(**actions)
        
        self.get_obs()

    def save(self):
        
        try:
            data_grp = self.f.create_group("data")
        except:
            data_grp = self.f["data"]        
        demo_grp = data_grp.create_group(f"demo_{self.demo_idx}")
        
        obs_grp = demo_grp.create_group("observations")
        for key in self.obs_keys:
            obs_grp.create_dataset(key, data=self.obs[key][:-1])
        act_grp = demo_grp.create_group("actions")
        for key in self.actions_keys:
            act_grp.create_dataset(key, data=self.actions[key])
        
        self.f.flush()
        self._reset_dicts()

    def __del__(self):
         self.f.close()