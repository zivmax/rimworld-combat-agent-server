import os
from dataclasses import dataclass, astuple
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import Box
from numpy.typing import NDArray
from torch.types import Tensor

from .memory import PrioritizedReplayBuffer
from .model import DQN


class DQNAgent:
    @dataclass
    class Transition:
        states: Tensor
        next_states: Tensor
        actions: Tensor
        rewards: Tensor
        done: Tensor

        def __iter__(self):
            return iter(astuple(self))

    def __init__(
        self,
        n_envs: int,
        obs_space: Box,
        act_space: Box,
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device
        self.n_envs = n_envs
        self.obs_space = obs_space
        self.act_space = act_space

        self.memory = PrioritizedReplayBuffer(capacity=300000, alpha=0.6)
        self.gamma = 0.85

        self.batch_size = 512
        self.learning_rate = 0.00015

        self.epsilon_start = 1.0
        self.epsilon_final = 0.001
        self.epsilon_decay = 0.999995

        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        self.steps = 0
        self.updates = 0
        self.explore = True

        self.policy_net = DQN(self.obs_space, self.act_space).to(device)
        self.target_net = DQN(self.obs_space, self.act_space).to(device)
        self.update_target_network()
        self.target_net.eval()
        self.target_net_update_freq = 3000

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # History tracking
        self.loss_history: List[float] = []
        self.q_value_history: List[float] = []
        self.td_error_history: List[float] = []
        self.eps_threshold_history: List[float] = []

    def remember(
        self,
        states: NDArray,
        next_states: NDArray,
        actions: NDArray,
        rewards: NDArray,
        dones: NDArray,
    ) -> None:
        for i in range(self.n_envs):
            state = torch.from_numpy(states[i]).to(self.device)
            next_state = torch.from_numpy(next_states[i]).to(self.device)
            action = torch.tensor(actions[i]).to(self.device)
            reward = torch.tensor(rewards[i], device=self.device)
            done = torch.tensor(dones[i], device=self.device)
            max_priority = (
                max(self.memory.priorities) if self.memory.priorities else 1.0
            )
            self.memory.push((state, next_state, action, reward, done), max_priority)

    def act(self, states: NDArray) -> NDArray:
        assert self.obs_space.contains(
            states[0]
        ), f"Invalid state: {states[0]} not in {self.obs_space}."

        self.steps += 1

        eps_threshold = max(
            self.epsilon_final,
            self.epsilon_start * (1 - np.exp(-5 * self.epsilon_decay**self.steps)),
        )
        self.eps_threshold_history.append(eps_threshold)

        batch_actions = np.zeros((self.n_envs, 2), dtype=self.act_space.dtype)
        explores = np.random.rand(self.n_envs) < eps_threshold
        self.explore = explores.any()

        # Handle random exploration
        random_indices = np.where(explores)[0]
        if len(random_indices) > 0:
            batch_actions[random_indices] = [
                self.act_space.sample() for _ in range(len(random_indices))
            ]

        # Handle exploitation
        if len(self.memory) >= self.batch_size:
            exploit_indices = np.where(~explores)[0]
            if len(exploit_indices) > 0:
                with torch.no_grad():
                    states_tensor = torch.from_numpy(states[exploit_indices]).to(
                        self.device
                    )
                    outputs = (
                        self.policy_net.forward(states_tensor).max(1)[1].cpu().numpy()
                    )
                    width = self.act_space.high[0] - self.act_space.low[0]

                    x = outputs % width + self.act_space.low[0]
                    y = outputs // width + self.act_space.low[0]

                    batch_actions[exploit_indices] = [
                        np.stack([x, y], axis=1, dtype=self.act_space.dtype)
                    ]

        return batch_actions

    def train(self) -> None:
        if self.explore or len(self.memory.buffer) < self.batch_size:
            return

        self.updates += 1

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        transitions, indices, weights = self.memory.sample(self.batch_size, self.beta)
        batch = self.Transition(*zip(*transitions))

        state_batch = torch.stack(batch.states)
        next_state_batch = torch.stack(batch.next_states)
        action_batch = torch.stack(batch.actions)
        reward_batch = torch.tensor(batch.rewards, device=self.device)
        done_batch = torch.tensor(batch.done, device=self.device)

        action_idx_batch: torch.Tensor = (
            action_batch[:, 0] - self.act_space.low[0]
        ) * self.act_space.high[0] + (action_batch[:, 1] - self.act_space.low[0])

        q_values_batch = self.policy_net.forward(state_batch)
        q_values_batch = q_values_batch.gather(
            1, action_idx_batch.long().unsqueeze(1)
        ).squeeze()

        with torch.no_grad():
            max_next_q_value_batch = (
                self.target_net.forward(next_state_batch).max(1)[0].detach()
            )

        target_value_batch = (
            reward_batch
            + torch.logical_not(done_batch) * max_next_q_value_batch * self.gamma
        )

        td_errors = q_values_batch - target_value_batch
        loss = (
            torch.tensor(weights, device=self.device)
            * F.smooth_l1_loss(q_values_batch, target_value_batch, reduction="none")
        ).mean()

        # Store history
        self.loss_history.append(loss.item())
        self.q_value_history.append(q_values_batch.mean().item())
        self.td_error_history.append(td_errors.mean().item())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, priorities)

        if self.updates % self.target_net_update_freq == 0:
            self.update_target_network()

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def draw_model(self, save_path: str = "./training_history.png") -> None:
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create a DataFrame with the training statistics
        stats_df = pd.DataFrame(
            {
                "Update": range(len(self.loss_history)),
                "Loss": self.loss_history,
                "Q-Value": self.q_value_history,
                "TD Error": self.td_error_history,
            }
        )

        # Create subplots for each metric
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Plot loss
        sns.lineplot(data=stats_df, x="Update", y="Loss", ax=ax1)
        ax1.set_title("Loss over Updates")

        # Plot Q-values
        sns.lineplot(data=stats_df, x="Update", y="Q-Value", ax=ax2)
        ax2.set_title("Q-Values over Updates")

        # Plot TD errors
        sns.lineplot(data=stats_df, x="Update", y="TD Error", ax=ax3)
        ax3.set_title("TD Error over Updates")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def draw_agent(self, save_path: str = "./epsilon_decay.png") -> None:
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create a DataFrame with the training statistics
        stats_df = pd.DataFrame(
            {
                "Step": range(len(self.eps_threshold_history)),
                "Epsilon": self.eps_threshold_history,
            }
        )

        # Create subplots for each metric
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot epsilon decay
        sns.lineplot(data=stats_df, x="Step", y="Epsilon", ax=ax)
        ax.set_title("Epsilon Decay over Steps")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
