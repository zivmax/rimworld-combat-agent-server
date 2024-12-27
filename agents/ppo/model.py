import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict
import os
import numpy as np
import torch.distributions as distributions


class ActorCritic(nn.Module):
    def __init__(self, obs_space: Box, act_space: Box) -> None:
        super(ActorCritic, self).__init__()
        self.obs_space = obs_space
        self.act_space = act_space[1]
        self.num_actions = (
            self.act_space.shape[0] * len(self.act_space.spaces)
            if isinstance(self.act_space, Dict)
            else self.act_space.shape[0]
        )

        # Convolutional layers
        self.conv = nn.Sequential(
            # First conv block: in_channels -> 32 channels
            nn.Conv3d(
                in_channels=obs_space.shape[0],
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            # Second conv block: 32 -> 64 channels
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            # Max pooling to reduce spatial dimensions by 2
            nn.MaxPool3d(2),
            # Third conv block: 64 -> 64 channels
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
        )

        # Calculate the size of the convolutional output
        dummy = torch.zeros(1, *obs_space.shape)
        conv_out = self.conv(dummy)
        conv_out_size = conv_out.view(conv_out.size(0), -1).size(1)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor):
        x = state.float()
        x = x.unsqueeze(0)
        x = x / (torch.tensor(self.obs_space.high, device=x.device)).repeat(
            x.shape[0], *(len(self.obs_space.high.shape) * [1])
        )
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        action_mean = self.actor(x)
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        state_values = self.critic(x)
        return action_mean, action_std, state_values

    def act(self, state: torch.Tensor):
        action_mean, action_std, state_values = self.forward(state)
        dist = distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action, action_log_prob, state_values

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        action_mean, action_std, state_values = self.forward(states)
        dist = distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action_log_probs, entropy, state_values

    def save(self, filepath: str) -> None:
        directory = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath))
        self.eval()
