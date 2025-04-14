from gymnasium import spaces, Env
import numpy as np
import csv
import threading
import tempfile
import onnxruntime
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.tcp_client_inference_env_runner import (
    TcpClientInferenceEnvRunner,
    _send_message,
    _get_message,
)
import base64
import gzip
import socket
import time
from ray.rllib.env.utils.external_env_protocol import RLlink as rllink
from ray.rllib.core import Columns
import torch as th


class MobNet(Env):
    def __init__(self, env_config=None):
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


def client_rl(port: int = 5555):
    def _set_state(msg_body):
        with tempfile.TemporaryDirectory():
            with open("_temp_onnx", "wb") as f:
                f.write(
                    gzip.decompress(
                        base64.b64decode(msg_body["onnx_file"].encode("utf-8"))
                    )
                )
                onnx_session = onnxruntime.InferenceSession("_temp_onnx")
                output_names = [o.name for o in onnx_session.get_outputs()]
        return onnx_session, output_names

    # Connect to server.
    while True:
        try:
            print(f"Trying to connect to localhost:{port} ...")
            sock_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_.connect(("localhost", port))
            break
        except ConnectionRefusedError:
            time.sleep(5)

    # Send ping-pong.
    _send_message(sock_, {"type": rllink.PING.name})
    msg_type, msg_body = _get_message(sock_)
    assert msg_type == rllink.PONG

    # Request config.
    _send_message(sock_, {"type": rllink.GET_CONFIG.name})
    msg_type, msg_body = _get_message(sock_)
    assert msg_type == rllink.SET_CONFIG
    env_steps_per_sample = msg_body["env_steps_per_sample"]
    force_on_policy = msg_body["force_on_policy"]

    # Request ONNX weights.
    _send_message(sock_, {"type": rllink.GET_STATE.name})
    msg_type, msg_body = _get_message(sock_)
    assert msg_type == rllink.SET_STATE
    onnx_session, output_names = _set_state(msg_body)

    # Episode collection buckets.
    episodes = []
    observations = []
    actions = []
    action_dist_inputs = []
    action_logps = []
    rewards = []

    timesteps = 0
    episode_return = 0.0

    # Start actual env loop.
    env = MobNet()
    obs, info = env.reset()
    observations.append(obs.tolist())

    while True:
        timesteps += 1
        # Perform action inference using the ONNX model.
        logits = onnx_session.run(
            output_names,
            {"onnx::Gemm_0": np.array([obs], np.float32)},
        )[0][
            0
        ]  # [0]=first return item, [0]=batch size 1

        # Stochastic sample.
        assert env.action_space.shape is not None, "Action space is not defined."
        mean = th.from_numpy(logits[0 : env.action_space.shape[0]])
        log_std = th.from_numpy(logits[env.action_space.shape[0] :])
        std = th.exp(log_std)
        dist = th.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action)
        squashed = th.tanh(action)  # Now in [-1, 1]
        # Scale to [0.1, 1.0]
        assert isinstance(env.action_space, spaces.Box), "Action space is not Box."
        low, high = env.action_space.low, env.action_space.high
        action_scaled = (squashed + 1) / 2 * (high - low) + low

        # Perform the env step.
        assert isinstance(action_scaled, th.Tensor), "Action is not a Torch tensor."
        obs, reward, terminated, truncated, info = env.step(action_scaled.numpy())

        # Collect step data.
        observations.append(obs.tolist())
        actions.append(action.tolist())
        action_dist_inputs.append(logits.tolist())
        action_logps.append(logp.tolist())
        rewards.append(reward)
        episode_return += reward

        # We have to create a new episode record.
        if timesteps == env_steps_per_sample or terminated or truncated:
            episodes.append(
                {
                    Columns.OBS: observations,
                    Columns.ACTIONS: actions,
                    Columns.ACTION_DIST_INPUTS: action_dist_inputs,
                    Columns.ACTION_LOGP: action_logps,
                    Columns.REWARDS: rewards,
                    "is_terminated": terminated,
                    "is_truncated": truncated,
                }
            )
            # We collected enough samples -> Send them to server.
            if timesteps == env_steps_per_sample:
                # Make sure the amount of data we collected is correct.
                assert sum(len(e["actions"]) for e in episodes) == env_steps_per_sample

                # Send the data to the server.
                if force_on_policy:
                    _send_message(
                        sock_,
                        {
                            "type": rllink.EPISODES_AND_GET_STATE.name,
                            "episodes": episodes,
                            "timesteps": timesteps,
                        },
                    )
                    # We are forced to sample on-policy. Have to wait for a response
                    # with the state (weights) in it.
                    msg_type, msg_body = _get_message(sock_)
                    assert msg_type == rllink.SET_STATE
                    onnx_session, output_names = _set_state(msg_body)

                # Sampling doesn't have to be on-policy -> continue collecting
                # samples.
                else:
                    raise NotImplementedError

                episodes = []
                timesteps = 0

            # Set new buckets to empty lists (for next episode).
            observations = [observations[-1]]
            actions = []
            action_dist_inputs = []
            action_logps = []
            rewards = []

            # The episode is done -> Reset.
            if terminated or truncated:
                obs, _ = env.reset()
                observations = [obs.tolist()]
                episode_return = 0.0


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
    tune.Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(  # type: ignore
            storage_path=ray_storage + agent_name,
            name=agent_name,
            checkpoint_config=tune.CheckpointConfig(num_to_keep=10),
            # stop=stop, TODO
        ),
    ).fit()


# Example usage
if __name__ == "__main__":
    server_thread = threading.Thread(target=server_rl)
    server_thread.start()
    client_rl()
