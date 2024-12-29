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
        obs_space: Box,
        act_space: Box,
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device: str = device
        self.obs_space: Box = obs_space
        self.act_space: Box = act_space

        self.memory = PrioritizedReplayBuffer(capacity=1000000, alpha=0.6)
        self.gamma: float = 0.85

        self.batch_size: int = 1024
        self.learning_rate: float = 0.00015

        self.epsilon_start: float = 1.0
        self.epsilon_final: float = 0.001
        self.epsilon_decay: float = 0.999995

        self.beta: float = 0.4
        self.beta_increment_per_sampling: float = 0.001

        self.steps: int = 0
        self.explore: bool = True

        self.policy_net: DQN = DQN(self.obs_space, self.act_space).to(device)
        self.target_net: DQN = DQN(self.obs_space, self.act_space).to(device)
        self.update_target_network()
        self.target_net.eval()
        self.target_net_update_freq = 3000

        self.optimizer: optim.Adam = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )

        # History tracking
        self.loss_history: List[float] = []
        self.q_value_history: List[float] = []
        self.td_error_history: List[float] = []

    def remember(
        self,
        state: NDArray,
        next_state: NDArray,
        action: NDArray,
        reward: float,
        done: bool,
    ) -> None:
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        action = torch.from_numpy(action).to(self.device)
        reward = torch.tensor(reward, device=self.device)
        done = torch.tensor(done, device=self.device)
        max_priority = max(self.memory.priorities) if self.memory.priorities else 1.0
        self.memory.push((state, next_state, action, reward, done), max_priority)

    def act(self, state: NDArray) -> Dict:
        self.steps += 1

        eps_threshold = max(
            self.epsilon_final,
            self.epsilon_start * (1 - np.exp(-5 * self.epsilon_decay**self.steps)),
        )

        self.explore = np.random.rand() < eps_threshold
        if self.explore or len(self.memory) < self.batch_size:
            return self.act_space.sample()
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).unsqueeze(0).to(self.device)
                output = self.policy_net.forward(state).max(1)[1].item()
                width = self.act_space.high[0] - self.act_space.low[0]
                x = output % width
                y = output // width

                x += self.act_space.low[0]
                y += self.act_space.low[0]

                return np.array([x, y])

    def train(self) -> None:
        if self.explore or len(self.memory.buffer) < self.batch_size:
            return

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        transitions, indices, weights = self.memory.sample(self.batch_size, self.beta)
        batch = self.Transition(*zip(*transitions))

        state_batch = torch.stack(batch.states)
        next_state_batch = torch.stack(batch.next_states)
        action_batch = torch.stack(batch.actions)
        reward_batch = torch.tensor(batch.rewards, device=self.device)
        done_batch = torch.tensor(batch.done, device=self.device)

        action_idx_batch = (
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

        if self.steps % self.target_net_update_freq == 0:
            self.update_target_network()

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def draw(self, save_path: str = "./training_history.png") -> None:
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create a DataFrame with the training statistics
        stats_df = pd.DataFrame(
            {
                "Step": range(len(self.loss_history)),
                "Loss": self.loss_history,
                "Q-Value": self.q_value_history,
                "TD Error": self.td_error_history,
            }
        )

        # Create subplots for each metric
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Plot loss
        sns.lineplot(data=stats_df, x="Step", y="Loss", ax=ax1)
        ax1.set_title("Loss over Time")

        # Plot Q-values
        sns.lineplot(data=stats_df, x="Step", y="Q-Value", ax=ax2)
        ax2.set_title("Q-Values over Time")

        # Plot TD errors
        sns.lineplot(data=stats_df, x="Step", y="TD Error", ax=ax3)
        ax3.set_title("TD Error over Time")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
