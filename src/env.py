import os

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

from gymnasium import spaces, Env
import numpy as np
import csv
import threading
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env.tcp_client_inference_env_runner import (
    TcpClientInferenceEnvRunner,
)


class MobNet(Env):
    def __init__(self):
        super(MobNet, self).__init__()
        self.steps_per_episode = 1000
        self.curr_step = 0
        self.curr_ep = 0
        self.max_number_ep = 10
        self.action_space = spaces.Box(low=0.1, high=1, shape=(2,))
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(2,), dtype=np.float32
        )
        self.ns3_config = {
            "number_slices": 2,
            "slice_ue_rnti": [[1, 2], [3, 4]],
        }

        self.slice_req = np.array([5, 10])

    def step(self, action):
        perc_action = np.floor((action / np.sum(action)) * 100)
        obs = self.observation_space.sample()  # TODO check this
        reward = self.calculate_reward(obs)
        terminated, truncated = False, False
        print(
            f"Episode: {self.curr_ep}, Step: {self.curr_step}, Reward: {reward} Action: {perc_action}, Obs: {obs}, Req: {self.slice_req}"
        )
        self.curr_step += 1
        if self.curr_step > self.steps_per_episode:
            terminated = truncated = True
            self.curr_ep += 1

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self.curr_step = 0
        if self.curr_ep > self.max_number_ep:
            self.curr_ep = 0
        return np.array([0, 0]), {}

    def calculate_reward(
        self,
        obs: np.ndarray,
    ) -> float:
        reward = 0.0
        under_thr = obs < self.slice_req
        if under_thr.any():
            reward -= np.mean(
                (self.slice_req[under_thr] - obs[under_thr]) / self.slice_req[under_thr]
            )
        assert isinstance(reward, float)
        return reward


def client_rl():
    env = MobNet()
    total_timesteps = int(1e9)
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=f"./tensorboard-logs/ppo/")
    model.learn(total_timesteps=total_timesteps)


def server_rl():
    ray_storage = "/tmp/ray_storage/"
    agent_name = "ppo"
    checkpoint_frequency = 1
    stop = {
        "episodes_total": 10,
    }
    env = MobNet()
    config = (
        PPOConfig()
        .environment(
            env=None,
            observation_space=env.observation_space,
            action_space=env.action_space,
            is_atari=False,
        )
        .framework("torch")
        .training(
            lr=0.0003,  # SB3 LR
            train_batch_size=2048,  # SB3 n_steps
            minibatch_size=64,  # type: ignore SB3 batch_size
            num_epochs=10,  # type: ignore SB3 n_epochs
            gamma=0.99,  # SB3 gamma
            lambda_=0.95,  # type: ignore # SB3 gae_lambda
            clip_param=0.2,  # type: ignore SB3 clip_range,
            vf_clip_param=np.inf,  # type: ignore SB3 equivalent to clip_range_vf=None
            entropy_coeff=0.01,  # type: ignore SB3 ent_coef
            vf_loss_coeff=0.5,  # type: ignore SB3 vf_coef
            grad_clip=0.5,  # SB3 max_grad_norm
            use_gae=True,  # type: ignore SB3 normalize_advantage
            kl_coeff=0,  # type: ignore
            use_kl_loss=False,  # type: ignore
            kl_target=0,  # type: ignore
        )
        .env_runners(
            env_runner_cls=TcpClientInferenceEnvRunner,
        )
    )
    results = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(  # type: ignore
            storage_path=ray_storage + agent_name,
            name=agent_name,
            checkpoint_config=train.CheckpointConfig(num_to_keep=10),
            # stop=stop, TODO
        ),
    ).fit()


# Example usage
if __name__ == "__main__":
    server_rl()
