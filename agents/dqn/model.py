import os

import torch.nn as nn
import torch
from gymnasium import spaces
import numpy as np


class DQN(nn.Module):
    def __init__(self, obs_space: spaces.Box, act_space: spaces.Box):
        """
        Initialize Deep Q Network

        Args:
            obs_shape (tuple): (height, width, channels)
            act_n (int): number of actions
        """
        super(DQN, self).__init__()
        self.obs_space = obs_space
        self.act_space = act_space

        self.conv = nn.Sequential(
            # First conv block: 6 -> 32 channels
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

        dummy = torch.zeros(1, *obs_space.shape)
        out = self.conv(dummy)
        conv_out_size = out.view(out.size(0), -1).size(1)

        # Calculate total number of actions from the Box space dimensions
        act_space_size = int(np.prod(act_space.high - act_space.low + 1))

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_size, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=act_space_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert the entire tensor to float
        x = x.float()

        x = x / (torch.tensor(self.obs_space.high, device=x.device)).repeat(
            x.shape[0], *(len(self.obs_space.high.shape) * [1])
        )

        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    def save(self, filepath: str) -> None:
        directory = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(
            torch.load(filepath, map_location=device, weights_only=True)
        )
