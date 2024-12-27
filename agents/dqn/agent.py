from dataclasses import dataclass
from typing import Dict
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, astuple
import torch
import torch.optim as optim
from .memory import PrioritizedReplayBuffer
from .model import DQN
import torch.nn.functional as F
from gymnasium.spaces import Box
from torch.types import Tensor


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
        self.target_net_update_freq = 300

        self.optimizer: optim.Adam = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )

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
