from typing import Tuple, Deque, Dict
import numpy as np
from numpy.typing import NDArray
from collections import namedtuple
import torch
import torch.optim as optim
from collections import deque
import random
from .model import DQN
import random
import torch.nn.functional as F
import math
from gymnasium.spaces import Space, Box, MultiDiscrete

Transition = namedtuple("Transition", ("state", "next_state", "action", "reward"))


class DQNAgent:
    def __init__(
        self,
        obs_space: Box,
        act_space: Dict[int, MultiDiscrete],
        device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device: str = device
        self.obs_space: Box = obs_space
        self.act_space: Dict[int, MultiDiscrete] = act_space
        self.memory: Deque[Tuple] = deque(maxlen=100000)
        self.batch_size: int = 32
        self.gamma: float = 0.99
        self.epsilon_max: float = 1.0
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.99950
        self.learning_rate: float = 0.00025
        self.dones: int = 0

        self.policy_net: DQN = DQN(self.obs_space, self.act_space[1]).to(device)
        self.target_net: DQN = DQN(self.obs_space, self.act_space[1]).to(device)
        self.update_target_network()
        self.target_net.eval()
        self.target_net_update_freq = 1000

        self.optimizer: optim.Adam = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )

    def remember(
        self,
        state: NDArray,
        next_state: NDArray,
        action: int,
        reward: float,
    ) -> None:
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        next_state = torch.from_numpy(next_state).unsqueeze(0).to(self.device)
        action = torch.tensor(np.array([action[1]]), device=self.device)
        reward = torch.tensor([reward], device=self.device)
        self.memory.append((state, next_state, action, reward))

    def act(self, state: NDArray) -> Dict:
        eps_threshold = self.epsilon_min + (
            self.epsilon_max - self.epsilon_min
        ) * math.exp(-1 * self.epsilon_decay**self.dones)

        self.dones += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.from_numpy(state).unsqueeze(0).to(self.device)
                output = self.policy_net.forward(state).max(1)[1].item()
                x = output // self.act_space[1].nvec[0]
                y = output % self.act_space[1].nvec[0]
                return {1: np.array([x, y])}
        else:
            return {1: self.act_space[1].sample()}

    def train(self) -> None:
        if len(self.memory) < self.batch_size * 100:
            return

        transitions = random.sample(self.memory, self.batch_size)

        batch = Transition(*zip(*transitions))

        actions = tuple(
            (map(lambda a: torch.tensor([[a]], device=self.device), batch.action))
        )
        rewards = tuple(
            (map(lambda r: torch.tensor([r], device=self.device), batch.reward))
        )

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.dones % self.target_net_update_freq == 0:
            self.update_target_network()

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
