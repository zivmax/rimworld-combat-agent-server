import os
from dataclasses import dataclass, astuple
from typing import Dict, List
from collections import deque

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
        """
        Dataclass to store transitions in the replay buffer.

        - `state0s`: the initial states
        - `stateNs`: the n step states
        - `action0s`: the initial actions
        - `rewardNs`: the n step rewards
        - `dones`: the done flags
        """

        state0s: Tensor
        stateNs: Tensor
        action0s: Tensor
        rewardNs: Tensor
        dones: Tensor

        def __iter__(self):
            return iter(astuple(self))

    def __init__(
        self,
        n_envs: int,
        obs_space: Box,
        act_space: Box,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device
        self.n_envs = n_envs
        self.obs_space = obs_space
        self.act_space = act_space

        self.memory = PrioritizedReplayBuffer(capacity=300000, alpha=0.6)
        self.gamma = 0.7

        self.batch_size = 1024
        self.learning_rate = 0.00015

        self.epsilon_start = 1.0
        self.epsilon_final = 0.001
        self.epsilon_decay = 0.999995

        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

        self.n_step = 4
        self.n_step_buffer = [deque(maxlen=self.n_step) for _ in range(self.n_envs)]
        self.gamma_n = self.gamma**self.n_step

        self.v_range = 150
        self.atoms = 102

        def create_dqn():
            net = DQN(
                self.obs_space,
                self.act_space,
                v_max=self.v_range,
                v_min=-self.v_range,
                atoms=self.atoms,
            ).to(device)
            return net

        self.supports = torch.linspace(-self.v_range, self.v_range, self.atoms)
        self.policy_net = create_dqn()
        self.target_net = create_dqn()
        self._update_target_network()
        self.target_net.eval()
        self.target_net_update_freq = 3000

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # History tracking
        self.loss_history: List[float] = []
        self.q_value_history: List[float] = []
        self.td_error_history: List[float] = []
        self.kl_div_history: List[float] = []
        self.eps_threshold_history: List[float] = []

        self.steps = 0
        self.updates = 0
        self.explore = True

    def remember(
        self,
        states: NDArray,
        next_states: NDArray,
        actions: NDArray,
        rewards: NDArray,
        dones: NDArray,
    ) -> None:
        for i in range(self.n_envs):
            state = torch.from_numpy(states[i]).cpu()
            next_state = torch.from_numpy(next_states[i]).cpu()
            action = torch.tensor(actions[i]).cpu()
            reward = torch.tensor(rewards[i]).cpu()
            done = torch.tensor(dones[i]).cpu()

            # N-step returns
            self.n_step_buffer[i].append((state, next_state, action, reward, done))
            if len(self.n_step_buffer[i]) < self.n_step:
                return

            # Calculate n-step return
            state0, _, action0, _, _ = self.n_step_buffer[i][0]
            stateN = next_state
            rewards_n = [transition[3] for transition in self.n_step_buffer[i]]

            # Get value estimate for final state
            next_state_value = self._get_next_act_value_estimate(stateN.to(self.device))

            return_n = self._compute_n_step_reward(rewards_n, next_state_value, done)

            # Store transition with n-step return
            max_priority = (
                max(self.memory.priorities) if self.memory.priorities else 1.0
            )
            self.memory.push((state0, stateN, action0, return_n, done), max_priority)

            self.n_step_buffer[i].popleft()

    def _get_next_act_value_estimate(self, state: Tensor) -> Tensor:
        # Distributional DQN value estimate
        with torch.no_grad():
            next_Q_dists = self.policy_net.forward(
                state.unsqueeze(0).to(self.device)
            ).cpu()
            next_action = self._get_expected_q_values(next_Q_dists).argmax()
            target_dists = self.target_net.forward(
                state.unsqueeze(0).to(self.device)
            ).cpu()
            return self._get_expected_q_values(target_dists)[next_action]

    def _compute_n_step_reward(self, rewards, next_value, done):
        n_step_reward = next_value
        for reward in reversed(rewards):
            n_step_reward = reward + self.gamma * n_step_reward * torch.logical_not(
                done
            )
        return n_step_reward

    def _get_expected_q_values(self, q_dist: Tensor) -> Tensor:
        assert q_dist.is_cpu, "Expected q_dist to be on CPU."
        return torch.sum(
            q_dist * self.policy_net.supports.view(1, 1, -1), dim=2
        ).squeeze()

    def act(self, states: NDArray) -> NDArray:
        states = np.array(states)

        assert self.obs_space.contains(
            states[0]
        ), f"Invalid state: {states[0]} not in {self.obs_space}."

        self.steps += self.n_envs

        eps_threshold = max(
            self.epsilon_final,
            self.epsilon_start * (1 - np.exp(-5 * self.epsilon_decay**self.steps)),
        )
        self.eps_threshold_history.extend([eps_threshold] * self.n_envs)

        batch_actions = np.zeros((self.n_envs, 2), dtype=self.act_space.dtype)
        self.explore = np.random.rand() < eps_threshold

        # Handle random exploration
        if self.explore:
            batch_actions = np.array(
                [self.act_space.sample() for _ in range(self.n_envs)]
            )

        # Handle exploitation using distributional Q-values
        elif len(self.memory) >= self.batch_size:
            with torch.no_grad():
                states_tensor = torch.from_numpy(states)
                # Get Q-value distributions - shape: (batch_size, n_actions, n_atoms)
                Q_dists = self.policy_net.forward(states_tensor.to(self.device)).cpu()

                expected_Qs = self._get_expected_q_values(Q_dists)

                # Get actions with highest expected Q-values
                raw_actions = expected_Qs.argmax(dim=1)

                # Convert to 2D coordinates
                width = self.act_space.high[0] - self.act_space.low[0] + 1
                x = raw_actions % width + self.act_space.low[0]
                y = raw_actions // width + self.act_space.low[1]
                batch_actions = np.column_stack([x, y]).astype(self.act_space.dtype)

        return batch_actions

    def train(self) -> None:
        if self.explore or len(self.memory.buffer) < self.batch_size:
            return

        for _ in range(self.n_envs):
            self.updates += 1

            self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

            # Sample from prioritized replay buffer
            transitions, indices, weights = self.memory.sample(
                self.batch_size, self.beta
            )
            batch = self.Transition(*zip(*transitions))

            # Stack batch elements
            states_batch = torch.stack(batch.state0s)
            stateNs_batch = torch.stack(batch.stateNs)
            action0s_batch = torch.stack(batch.action0s)
            rewardNs_batch = torch.stack(batch.rewardNs)
            dones_batch = torch.stack(batch.dones)

            # Convert actions to indices
            action_idx_batch: torch.Tensor = (
                action0s_batch[:, 0] - self.act_space.low[0]
            ) * self.act_space.high[0] + (action0s_batch[:, 1] - self.act_space.low[0])

            # Get Q-values for initial state-action pairs
            q_dist_batch = self.policy_net.forward(states_batch.to(self.device)).cpu()
            expected_q = torch.sum(
                q_dist_batch * self.policy_net.supports.view(1, 1, -1), dim=2
            )
            q_dist_batch = expected_q.gather(
                1, action_idx_batch.long().unsqueeze(1)
            ).squeeze()

            with torch.no_grad():
                next_dist = self.target_net.forward(stateNs_batch.to(self.device))
                next_action = (
                    self.policy_net.forward(stateNs_batch.to(self.device))
                    .mean(dim=2)
                    .argmax(dim=1)
                )
                next_dist = next_dist[torch.arange(next_dist.size(0)), next_action, :]
                # Project the distribution using the supports, v_min, v_max, delta_z
                target_dist_batch = self._project_distribution(
                    next_dist,
                    rewardNs_batch,
                    dones_batch,
                )

            # Calculate TD errors and loss
            td_errors = (q_dist_batch - target_dist_batch).to(self.device)
            loss = (
                torch.tensor(weights, device=self.device)
                * F.smooth_l1_loss(q_dist_batch, target_dist_batch, reduction="none")
            ).mean()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.optimizer.step()

            # Update priorities using KL divergence
            kl_div = F.kl_div(
                F.log_softmax(q_dist_batch, dim=1),
                F.softmax(target_dist_batch, dim=1),
                reduction="none",
            ).sum(dim=1)
            priorities = kl_div.abs().detach().cpu().numpy() + 1e-5
            self.memory.update_priorities(indices, priorities)

            # Store history
            self.loss_history.append(loss.item())
            self.q_value_history.append(q_dist_batch.mean().item())
            self.td_error_history.append(td_errors.mean().item())
            self.kl_div_history.append(kl_div.mean().item())

            # Periodically update target network and reset noise
            if self.updates % self.target_net_update_freq == 0:
                self._update_target_network()
                self.policy_net.reset_noise()
                self.target_net.reset_noise()

    def _update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _project_distribution(self, next_dist: Tensor, rewards: Tensor, dones: Tensor):
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        supports = self.supports.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * self.gamma_n * supports
        Tz = Tz.clamp(min=self.policy_net.v_min, max=self.policy_net.v_max)
        b = (Tz - self.policy_net.v_min) / self.policy_net.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        projected_dist = torch.zeros_like(next_dist)
        projected_dist.view(-1).index_add_(
            0,
            (l + self.policy_net.supports.size(0) * torch.arange(self.batch_size)),
            (next_dist * (u.float() - b)).view(-1),
        )
        projected_dist.view(-1).index_add_(
            0,
            (u + self.policy_net.supports.size(0) * torch.arange(self.batch_size)),
            (next_dist * (b - l.float())).view(-1),
        )

        return projected_dist

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
                "KL Divergence": self.kl_div_history,
            }
        )

        # Create subplots for each metric
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))

        # Plot loss
        sns.lineplot(data=stats_df, x="Update", y="Loss", ax=ax1)
        ax1.set_title("Loss over Updates")

        # Plot Q-values
        sns.lineplot(data=stats_df, x="Update", y="Q-Value", ax=ax2)
        ax2.set_title("Q-Values over Updates")

        # Plot TD errors
        sns.lineplot(data=stats_df, x="Update", y="TD Error", ax=ax3)
        ax3.set_title("TD Error over Updates")

        # Plot KL divergence
        sns.lineplot(data=stats_df, x="Update", y="KL Divergence", ax=ax4)
        ax4.set_title("KL Divergence over Updates")

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
