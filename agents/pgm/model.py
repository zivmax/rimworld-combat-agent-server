import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict
import os
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, obs_space: Box, act_space: Dict) -> None:
        super(PolicyNetwork, self).__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.act_size = int(np.prod(self.act_space.high - self.act_space.low + 1))
        self.dim_actions = (
            self.act_space.shape[0] * len(self.act_space.spaces)
            if isinstance(self.act_space, Dict)
            else self.act_space.shape[0]
        )

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=obs_space.shape[0],
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
        )

        dummy = torch.zeros(1, *obs_space.shape)
        conv_out = self.conv(dummy)
        conv_out_size = conv_out.view(conv_out.size(0), -1).size(1)

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.dim_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.dim_actions))

    def forward(self, state: torch.Tensor):
        x = state.float()
        x = x.unsqueeze(0) if len(x.shape) != 5 else x
        x = x / (torch.tensor(self.obs_space.high, device=x.device)).repeat(
            x.shape[0], *(len(self.obs_space.high.shape) * [1])
        )
        x = self.conv.forward(x)
        x = x.view(x.size(0), -1)  # Flatten
        action_mean = self.actor.forward(x)
        action_mean = action_mean * self.act_size
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        action_mean = action_mean.view(-1, 2, self.dim_actions // 2)
        return action_mean, action_std

    def save(self, filepath: str) -> None:
        directory = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath))
        self.eval()
