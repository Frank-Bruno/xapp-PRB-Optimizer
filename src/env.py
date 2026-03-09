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
from .llm_reward import LLMAgent
from collections import deque
import csv
from datetime import datetime

class MobNet(Env):
    def __init__(self, slice_number=2, save_energy=False, env_config=None, debug=False, llm_mode=True):
        super(MobNet, self).__init__()
        self.llm_agent = LLMAgent()
        self.intent = """
            Considere uma situação onde tem-se dois slices de rede, cada um com seus próprios requisitos de desempenho. Os requisitos para cada slice são: 10 e 2 Mbps, respectivamente.
            Deseja-se cumprir os requisitos e não há interesse em economia de recursos.
            """
        #self.case_num = 
        self.save_energy = self.llm_agent.get_classification_prompt(self.intent)
        
        self.steps_per_episode = 128
        self.curr_step = 0
        self.curr_ep = 0
        self.max_number_ep = 10
        self.slice_number = slice_number
        #self.save_energy = save_energy
        self.action_space = spaces.Box(low=0.1, high=1, shape=(self.slice_number + int(bool(self.save_energy)),))
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.slice_number*2,), dtype=np.float32
        )
        self.max_rbs_rate = 1.0

        #self.slice_req = np.array([4, 1, 0.1, 0.001])
        self.slice_req = self.llm_agent.get_requiriments(self.intent)
        self.debug = debug
        self.k = self.llm_agent.k_type(self.intent, self.slice_req)
        self.buffer = 1
        
        ## PARA O PROPORTIONAL FAIR
        self.obs_window = 10

        self.ewma_alpha = 0.3
        self.slice_obs_hist = [deque(maxlen=self.obs_window) for _ in range(self.slice_number)]

        self.slice_obs_avg = np.zeros(self.slice_number, dtype=np.float64)
        self.llm_mode = llm_mode
        
        self.csv_file = "caso_1_log.csv"
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp","episode", "step", "reward", "max_rbs_rate", "action", "obs", "slice_req"
                ])
            
        #1°Caso
        #    """
        #    Considere uma situação onde tem-se dois slices de rede, cada um com seus próprios requisitos de desempenho.
        #    Deseja-se cumprir os requisitos e não há interesse em economia de recursos.
        #    """
        #2°Caso
        #    """
        #    Considere uma situação onde tem-se dois slices de rede, cada um com seus próprios requisitos de desempenho.
        #    Deseja-se cumprir os requisitos do primeiro slice, sem ultrapassá-los ou ficar abaixo, para maximizar o desempenho no segundo slice, no qual é desejado que o desempeno seja o maior possível.
        #    """
        #3°Caso
        #    """
        #    Considere uma situação onde tem-se dois slices de rede, cada um com seus próprios requisitos de desempenho.
        #    Deseja-se cumprir os requisitos, mas há interesse em economia de recursos.
        #    """
        #4°Caso
        #    """
        #    Considere uma situação na qual tem-se dois slices de rede, com seus requisitos de performance sendo vazão de dados e latência, respectivamente.
        #    Deseja-se otimizar a performance dos dois slices simultaneamente.
        #    """
    def step(self, action):
        if self.llm_mode:
            reward = self.calculate_reward(self.obs)
        else:
            reward = 1
        terminated, truncated = False, False
        
        timestamp = datetime.utcnow().isoformat()
        with open(self.csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, 
                self.curr_ep,
                self.curr_step,
                reward,
                self.max_rbs_rate,
                action.tolist() if hasattr(action, "tolist") else action,
                self.obs.tolist() if hasattr(self.obs, "tolist") else self.obs,
                np.asarray(self.slice_req).tolist() if hasattr(np.asarray(self.slice_req), "tolist") else self.slice_req
            ])
        
        if self.debug:
            print(
                f"DEBUG: Episode: {self.curr_ep}, Step: {self.curr_step}, Reward: {reward}, Max_BRs: {self.max_rbs_rate}, Action: {action}, Obs: {self.obs}, Req: {self.slice_req}"
            )
        self.curr_step += 1
        if self.curr_step > self.steps_per_episode:
            terminated = truncated = True
            self.curr_ep += 1

        return self.obs, reward, terminated, truncated, {}
    
    def generate_action(self, action):
        if self.save_energy:
            max_rbs_rate = action[-1]
            self.max_rbs_rate = max_rbs_rate
        else:
            max_rbs_rate = self.max_rbs_rate
        rbs_action = action[0:self.slice_number]
        perc_action = np.floor((rbs_action / np.sum(rbs_action)) * (100*max_rbs_rate))
        return perc_action

    def reset(self, seed=None, options=None):
        self.curr_step = 0
        if self.curr_ep > self.max_number_ep:
            self.curr_ep = 0
        return np.zeros((self.slice_number*2)), {}

    def set_obs(self, obs):
        self.obs = obs
        obs = obs if self.curr_step > 0 else np.ones((self.slice_number*2))
        if self.llm_mode:
            self.record_network_observation(obs)

    def calculate_reward(
        self,
        slice_obs: np.ndarray,
    ) -> float:
        
        if not self.llm_agent.existing_code():
            #print("O código ainda não existe ou não foi encontrado.")
            #TODO: Corrigir para considerar todos os casos
            code = self.llm_agent.create_reward_function(self.intent,self.slice_req[:int(np.size(self.slice_req)/2)], self.k)
            reward = self.llm_agent.run_reward_function(slice_obs[:int(np.size(slice_obs)/2)], self.slice_req[:int(np.size(self.slice_req)/2)], self.buffer, self.k)
            
            # Para o caso 4
            #reward = self.llm_agent.run_reward_function(slice_obs, self.slice_req, self.k) 
        else:
            reward = self.llm_agent.run_reward_function(slice_obs[:int(np.size(slice_obs)/2)], self.slice_req[:int(np.size(self.slice_req)/2)], self.buffer, self.k)
            
            # Para o caso 4
            #reward = self.llm_agent.run_reward_function(slice_obs, self.slice_req, self.buffer, self.k) 
        #reward -= np.mean(
        #    (self.slice_req[under_thr] - obs[under_thr]) / self.slice_req[under_thr]
        #)
        assert isinstance(reward, float)
        return reward
    
    def proportional_fair_allocation(self, slice_obs, slice_obs_avg):
        proportion = slice_obs / slice_obs_avg

        allocation = proportion / proportion.sum()

        return allocation
    
    def proportional_fair_schedule(self):
        thr = np.array(self.obs[:self.slice_number], dtype=np.float64)
        avg = np.where(self.slice_obs_avg > 0, self.slice_obs_avg, np.maximum(thr, 1e-9))
        proportion = thr / avg
        proportion[proportion <= 0] = 1e-9
        allocation = proportion / proportion.sum()
        #tau = 2
        #exp_metrics = np.exp(proportion/tau)
        #allocation = exp_metrics / exp_metrics.sum()
        return allocation *100
    
    def record_network_observation(self, slice_obs: np.ndarray):
        thr = np.array(slice_obs[:self.slice_number], dtype=np.float64)
        for i, val in enumerate(thr):
            self.slice_obs_hist[i].append(val)
            if self.slice_obs_avg[i] == 0 and len(self.slice_obs_hist[i]) == 1:
                self.slice_obs_avg[i] = val
            else:
                #self.slice_obs_avg[i] = (1 - self.ewma_alpha) * self.slice_obs_avg[i] + self.ewma_alpha * val
                self.slice_obs_avg[i] = np.mean(list(self.slice_obs_hist[i]))
        #print(f"DEBUG: Slice hist: {self.slice_obs_hist}")
        #print(f"DEBUG: Slice Obs Avg:{self.slice_obs_avg}")    
        return self.slice_obs_avg


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


def server_rl(slice_number=2, save_energy=False, env_config=None, debug=False, llm_mode=True):
    SERVER_ADDRESS = "localhost"
    SERVER_BASE_PORT = 9900
    RAY_STORAGE = "./ray_results/"
    AGENT_NAME = "oai_ppo"
    EPISODES_TOTAL = 100000
    DEBUG_MODE = False
    env = MobNet(slice_number=slice_number, save_energy=save_energy, debug=debug, llm_mode=llm_mode)
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
            train_batch_size=128,  # SB3 n_steps
            sgd_minibatch_size=32,  # type: ignore SB3 batch_size
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
