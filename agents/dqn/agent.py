from dataclasses import dataclass
from typing import Tuple, Deque, Dict
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, astuple
import torch
import torch.optim as optim
from collections import deque
import random
from .model import DQN
import random
import torch.nn.functional as F
import math
from gymnasium.spaces import Box
from torch.types import Tensor


@dataclass
class Transition:
    states: Tensor
    next_states: Tensor
    actions: Tensor
    rewards: Tensor
    done: Tensor

    def __iter__(self):
        return iter(astuple(self))


class DQNAgent:
    def __init__(
        self,
        obs_space: Box,
        act_space: Box,
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device: str = device
        self.obs_space: Box = obs_space
        self.act_space: Box = act_space
        self.memory: Deque[Tuple] = deque(maxlen=100000)
        self.batch_size: int = 32
        self.gamma: float = 0.99
        self.epsilon_final: float = 1.0
        self.epsilon_start: float = 0.01
        self.epsilon_decay: float = 0.99989
        self.learning_rate: float = 0.00015
        self.steps: int = 0

        self.policy_net: DQN = DQN(self.obs_space, self.act_space).to(device)
        self.target_net: DQN = DQN(self.obs_space, self.act_space).to(device)
        self.update_target_network()
        self.target_net.eval()
        self.target_net_update_freq = 200

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
        self.memory.append((state, next_state, action, reward, done))

    def act(self, state: NDArray) -> Dict:
        eps_threshold = self.epsilon_start + (
            self.epsilon_final - self.epsilon_start
        ) * math.exp(-5 * self.epsilon_decay**self.steps)

        self.steps += 1

        if random.random() < eps_threshold:
            with torch.no_grad():
                state = torch.from_numpy(state).unsqueeze(0).to(self.device)
                output = self.policy_net.forward(state).max(1)[1].item()
                x = output // self.act_space.high[0] + self.act_space.low[0]
                y = output % self.act_space.high[0] + self.act_space.low[0]
                return np.array([x, y])
        else:
            return self.act_space.sample()

    def train(self) -> None:
        if len(self.memory) < self.memory.maxlen:
            return

        transitions = random.sample(self.memory, self.batch_size)

        # Extract different attributes from transitions
        batch = Transition(*zip(*transitions))

        # Stack the tensors into the batches
        state_batch = torch.stack(batch.states)
        next_state_batch = torch.stack(batch.next_states)
        action_batch = torch.stack(batch.actions)
        reward_batch = torch.tensor(batch.rewards, device=self.device)
        done_batch = torch.tensor(batch.done, device=self.device)

        # Encode the actions back into the indices of the networks output
        action_idx_batch = (
            action_batch[:, 0] - self.act_space.low[0]
        ) * self.act_space.high[0] + (action_batch[:, 1] - self.act_space.low[0])

        # Get Q-values for all actions in current state
        q_values_batch = self.policy_net.forward(state_batch)
        q_values_batch = q_values_batch.gather(
            1, action_idx_batch.long().unsqueeze(1)
        ).squeeze()

        # Get Max Q-values we could get in next state
        with torch.no_grad():
            max_next_q_value_batch = (
                self.target_net.forward(next_state_batch).max(1)[0].detach()
            )

        target_value_batch = (
            reward_batch
            + torch.logical_not(done_batch) * max_next_q_value_batch * self.gamma
        )

        loss = F.smooth_l1_loss(q_values_batch, target_value_batch)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.steps % self.target_net_update_freq == 0:
            self.update_target_network()

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
