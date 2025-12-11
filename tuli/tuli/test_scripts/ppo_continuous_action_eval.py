from typing import Callable

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
import robosuite
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from scipy.fftpack import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from tuli.envs import WipeSphere
from tuli.utils.viz_utils import plot_peak_freq

# Register the environment
robosuite.environments.REGISTERED_ENVS["WipeSphere"] = WipeSphere


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def make_robosuite_env(idx, capture_video, run_name, gamma, task=None):
    import robosuite
    from robosuite.wrappers import GymWrapper
    from robosuite.controllers import load_composite_controller_config

    def thunk():
        # controller_config = load_composite_controller_config(controller="BASIC")
        controller_config = load_composite_controller_config(robot="Panda")
        env = robosuite.make(
            "Wipe",
            robots=["Panda"],             # load a Sawyer robot and a Panda robot
            gripper_types="SphereGripper",                # use default grippers per robot arm
            controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
            has_renderer=True,                     # no on-screen rendering
            has_offscreen_renderer=False,           # no off-screen rendering
            control_freq=20,                        # 20 hz control for applied actions
            horizon=1000,                            # each episode terminates after 200 steps
            use_object_obs=True,                    # provide object observations to agent
            use_camera_obs=False,                   # don't provide image observations to agent
            reward_shaping=True,                    # use a dense reward signal for learning
        )
        env = GymWrapper(env, keys=[
                            'robot0_joint_pos_cos', 
                            'robot0_joint_pos_sin', 
                            'robot0_joint_vel',
                            'robot0_eef_pos',
                            # 'puck_pos',
                            # 'puck_goal_dist',
                            # 'goal_pos',
                        ])
        
        # Add metadata to the environment
        env.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 20,
            "semantics.async": False
        }

        capture_video = True
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, capture_video, run_name, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    robot = envs.envs[0].robots[0]

    obs, _ = envs.reset()
    episodic_returns = []

    # for computing peak frequencies
    force_history_horizon = 100
    force_history = []
    all_peak_freqs = []
    timestep = 0

    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        # breakpoint()
        robot = envs.envs[0].robots[0]
        # print("ee_vel: ", np.linalg.norm(robot._hand_total_velocity["right"][:3]), np.linalg.norm(robot.ee_force["right"]))
        
        # ===== for computing peak frequencies =====
        ee_force = np.linalg.norm(robot.ee_force["right"])
        force_history.append(ee_force)
        timestep += 1
        
        start_idx = timestep - force_history_horizon

        if start_idx > 0:
            force_values = np.array(force_history[start_idx:])
            # Compute FFT
            N = force_history_horizon
            freqs = np.fft.fftfreq(N, d=1)  # Frequency axis
            fft_values = np.abs(fft(force_values))  # Magnitude of FFT

            # Find peaks in the frequency domain
            peaks, _ = find_peaks(fft_values, height=0.1 * max(fft_values))
            peak_freqs = freqs[peaks]
            print("peak_freqs: ", peak_freqs)
            all_peak_freqs.append(peak_freqs)
        # ===== for computing peak frequencies =====
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
                plot_peak_freq(force_history, all_peak_freqs)
                force_history = []
                all_peak_freqs = [] 
                force_history = []
                timestep = 0
                # breakpoint()
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    # from tuli.test_scripts.ppo_continuous_action import Agent, make_robosuite_env

    model_path = f"/home/arpit/test_projects/tuli/runs/wipe_sensory_reward_6dof/ppo_continuous_action.cleanrl_model"

    episodic_returns = evaluate(
            model_path,
            make_robosuite_env,
            env_id="wipe",
            eval_episodes=10,
            run_name=f"eval",
            Model=Agent,
            device="cpu",
            gamma=0.99,
        )
