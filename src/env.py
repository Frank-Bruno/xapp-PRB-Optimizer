import argparse
from pathlib import Path

import numpy as np
from gymnasium import Env, spaces
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.policy_client import PolicyClient
import argparse
import os
from math import inf
from pathlib import Path
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.registry import get_trainable_cls
import threading
from time import sleep


class MobNet(Env):
    def __init__(self, env_config=None, debug=False):
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
        self.debug = debug

    def step(self, action):
        perc_action = np.floor((action / np.sum(action)) * 100)
        reward = self.calculate_reward(self.obs)
        terminated, truncated = False, False
        if self.debug:
            print(
                f"Episode: {self.curr_ep}, Step: {self.curr_step}, Reward: {reward} Action: {perc_action}, Obs: {self.obs}, Req: {self.slice_req}"
            )
        self.curr_step += 1
        if self.curr_step > self.steps_per_episode:
            terminated = truncated = True
            self.curr_ep += 1

        return self.obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self.curr_step = 0
        if self.curr_ep > self.max_number_ep:
            self.curr_ep = 0
        return np.array([0, 0]), {}

    def set_obs(self, obs):
        self.obs = obs

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


def client_rl(test_mode: bool = False):
    SERVER_ADDRESS = "localhost"
    RAY_STORAGE = "./ray_results/"
    AGENT_NAME = "ppo"
    SERVER_BASE_PORT = 9900
    env = MobNet()

    # Start a new episode.
    obs, info = env.reset()

    if test_mode:  # Testing
        ray_storage = str(Path(RAY_STORAGE).resolve())
        analysis = tune.ExperimentAnalysis(f"{ray_storage}/{AGENT_NAME}/")
        assert analysis.trials is not None, "Analysis trial is None"
        last_checkpoint = analysis.get_last_checkpoint(analysis.trials[0])
        assert last_checkpoint is not None, "Last checkpoint is None"
        algo = Algorithm.from_checkpoint(last_checkpoint)
        while True:
            try:
                action = algo.compute_single_action(obs, explore=False)
                assert isinstance(action, np.ndarray), "Action must be a numpy array."
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    _, _ = env.reset()
            except KeyboardInterrupt:
                break
    else:  # Training
        client = PolicyClient(
            f"http://{SERVER_ADDRESS}:{SERVER_BASE_PORT}",
            inference_mode="local",
        )
        eid = client.start_episode(training_enabled=True)
        while True:
            try:
                action = client.get_action(eid, obs)

                assert isinstance(action, np.ndarray), "Action must be a numpy array."
                obs, reward, terminated, truncated, info = env.step(action)

                # Log next-obs, rewards, and infos.
                client.log_returns(eid, reward, info=info)

                # Reset the episode if done.
                if terminated or truncated:
                    client.end_episode(eid, obs)
                    obs, info = env.reset()
                    eid = client.start_episode(training_enabled=True)
            except KeyboardInterrupt:
                break


def server_rl():
    SERVER_ADDRESS = "localhost"
    SERVER_BASE_PORT = 9900
    RAY_STORAGE = "./ray_results/"
    AGENT_NAME = "oai_ppo"
    EPISODES_TOTAL = 100000
    DEBUG_MODE = False
    env = MobNet()

    if __name__ == "__main__":
        ray_storage = str(Path(RAY_STORAGE).resolve())
        ray.init(local_mode=DEBUG_MODE)

        def _input(ioctx):
            if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
                return PolicyServerInput(
                    ioctx,
                    SERVER_ADDRESS,
                    SERVER_BASE_PORT
                    + ioctx.worker_index
                    - (1 if ioctx.worker_index > 0 else 0),
                )
            else:
                return None

    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        .environment(
            env=None,
            observation_space=env.observation_space,
            action_space=env.action_space,
            is_atari=False,
        )
        .framework("torch")
        .offline_data(input_=_input)
        .rollouts(
            num_rollout_workers=0,
            enable_connectors=False,
        )
        .evaluation(off_policy_estimation_methods={})
        .debugging(log_level="INFO")
        .training(
            lr=0.0003,  # SB3 LR
            train_batch_size=2048,  # SB3 n_steps
            sgd_minibatch_size=64,  # type: ignore SB3 batch_size
            num_sgd_iter=10,  # type: ignore SB3 n_epochs
            gamma=0.99,  # SB3 gamma
            lambda_=0.95,  # type: ignore # SB3 gae_lambda
            clip_param=0.2,  # type: ignore SB3 clip_range,
            vf_clip_param=np.inf,  # type: ignore SB3 equivalent to clip_range_vf=None
            use_gae=True,  # type: ignore SB3 normalize_advantage
            entropy_coeff=0.01,  # type: ignore SB3 ent_coef
            vf_loss_coeff=0.5,  # type: ignore SB3 vf_coef
            grad_clip=0.5,  # SB3 max_grad_norm TODO
            # kl_target=0.00001,  # SB3 target_kl
        )
        # .rl_module(_enable_rl_module_api=False)
        # .experimental(_enable_new_api_stack=False)
    )

    config["model"]["fcnet_hiddens"] = [
        64,
        64,
    ]  # Set neural network size

    config.experimental()
    config.update_from_dict(
        {
            "train_batch_size": 1000,
            # "model": {"use_lstm": args.use_lstm},
        }
    )

    stop = {
        "episodes_total": EPISODES_TOTAL,
    }

    # Restore the agent training in case of interruption or starts a new training
    if tune.Tuner.can_restore(f"{ray_storage}/{AGENT_NAME}/"):
        tuner = tune.Tuner.restore(
            f"{ray_storage}/{AGENT_NAME}/", trainable=PPO, param_space=config
        )
        results = tuner.fit()
    else:
        results = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=air.RunConfig(
                stop=stop,
                verbose=2,
                storage_path=ray_storage,
                name=AGENT_NAME,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=1,
                    checkpoint_at_end=True,
                ),
            ),
        ).fit()


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client",
        action="store_true",
    )
    parser.add_argument(
        "--test",
        action="store_true",
    )
    parser.add_argument(
        "--server",
        action="store_true",
    )
    args = parser.parse_args()
    if args.client:
        client_rl(test_mode=args.test)
    elif args.server:
        server_rl()
    else:
        server_thread = threading.Thread(target=server_rl)
        server_thread.start()
        sleep(10)
        client_rl()
        # Reverse
        # client_thread = threading.Thread(target=client_rl)
        # client_thread.start()
        # server_rl()
