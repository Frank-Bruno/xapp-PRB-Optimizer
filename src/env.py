from gymnasium import spaces, Env
import numpy as np
import csv
import threading
from stable_baselines3.ppo.ppo import PPO
from src.aux_xapp_rl import AuxXappRl

class MobNet(Env):
    def __init__(self):
        super(MobNet, self).__init__()
        self.obs_file = "/tmp/src/obs.csv"
        self.action_file = "/tmp/src/action.csv"
        self.steps_per_episode = 1000
        self.curr_step = 0
        self.curr_ep = 0
        self.max_number_ep = 10
        self.action_space = spaces.Box(low=0.1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.ns3_config = {
            "number_slices": 2,
            "slice_ue_rnti": [[1, 2], [3, 4]],
        }

        self.slice_req = np.array([5, 10])


    def step(self, action):
        perc_action = self.apply_action(action)
        obs = self.calc_obs()
        reward = self.calculate_reward(obs)
        terminated, truncated = False, False
        print(f"Episode: {self.curr_ep}, Step: {self.curr_step}, Reward: {reward} Action: {perc_action}, Obs: {obs}, Req: {self.slice_req}")
        self.curr_step += 1
        if self.curr_step > self.steps_per_episode:
            terminated = truncated = True
            self.curr_ep += 1

        return obs, reward, terminated, truncated, {}
    
    def apply_action(self, action):
        perc_action = np.floor((action/np.sum(action)) * 100)
        with open(self.action_file, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(perc_action.tolist())
        
        return perc_action
    
    def calc_obs(self):
        curr_ues_thr = np.zeros(2)
        valid_obs = False
        while not valid_obs:
            with open(self.obs_file, mode='r') as file:
                csv_reader = csv.reader(file)
                for i, row in enumerate(csv_reader):
                    if i == 0:
                        curr_ues_thr = np.array([float(value) for value in row[:2]])
                        valid_obs = True
                        break
            if valid_obs:
                with open(self.obs_file, mode='w') as file:
                    file.truncate()
        return curr_ues_thr

    def reset(self, seed=None, options=None):
        self.curr_step = 0
        if self.curr_ep > self.max_number_ep:
            self.curr_ep = 0
        with open(self.obs_file, mode='w') as file:
            file.truncate()
        return np.array([0,0]), {}
    
    def calculate_reward(
        self, obs: np.ndarray,
    ) -> float:
        reward = 0.0
        under_thr = obs < self.slice_req
        if under_thr.any():
            reward -= np.mean((self.slice_req[under_thr] - obs[under_thr])/self.slice_req[under_thr])
        assert isinstance(reward, float)
        return reward
    
def train_sb3():
    env = MobNet()
    total_timesteps = int(1e9)
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=f"./tensorboard-logs/ppo/")
    model.learn(total_timesteps=total_timesteps)

# Example usage
if __name__ == "__main__":
    fake_xapp_rl = AuxXappRl()
    env_thread = threading.Thread(target=fake_xapp_rl.run, daemon=True)
    env_thread.start()
    train_sb3()