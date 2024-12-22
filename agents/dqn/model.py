import random
import numpy as np
from collections import deque
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from .hyper_params import EPSILON, TARGET_UPDATE, MEMORY_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2
from .hyper_params import BATCH_X, BATCH_SIZE, GAMMA, LEARNING_RATE, DEVICE
import os


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size1=HIDDEN_SIZE1,
                 hidden_size2=HIDDEN_SIZE2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DQNModel:
    def __init__(
        self,
        state_size,
        action_size,
        action_space,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        epsilon_start=EPSILON["START"],
        epsilon_end=EPSILON["FINAL"],
        epsilon_decay=EPSILON["DECAY"],
        target_update=TARGET_UPDATE,
        memory_size=MEMORY_SIZE,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.memory = deque(maxlen=memory_size)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.target_update = target_update
        self.steps_done = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        # exploitation
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def _index_to_action(self, index: int) -> Dict[int, Tuple[int, int]]:
        """
        Convert a flat action index back to a structured action per ally.

        Args:
            index (int): Flat action index.

        Returns:
            Dict[int, Tuple[int, int]]: Structured action dictionary.
        """
        for _, space in self.action_space.items():
            n = space.nvec.prod()
            action_part = index % n
            index = index // n
        return int(action_part // space.nvec[1]), int(action_part % space.nvec[1])

    # memory sample estimation
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)

        # Q value iteration
        acts = []
        for action in actions:
            x, y = self._index_to_action(action)
            act_policy_index = x + y * BATCH_X
            acts.append(act_policy_index)
        acts = torch.LongTensor(acts).unsqueeze(1).to(self.device)
        current_q = self.policy_net(states).gather(1, acts)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # backward
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        """
        Save the model parameters to the specified path.

        Args:
            path (str): Path to save the model parameters.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        """
        Load the model parameters from the specified path.

        Args:
            path (str): Path to load the model parameters from.
        """
        self.policy_net.load_state_dict(torch.load(path))
