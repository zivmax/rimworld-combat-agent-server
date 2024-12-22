import torch.nn as nn
import torch
import os
from gymnasium import spaces
from typing import Dict


class DQN(nn.Module):
    def __init__(self, obs_space: spaces.Box, act_space: spaces.MultiDiscrete):
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
            nn.Conv2d(
                in_channels=obs_space.shape[0], out_channels=32, kernel_size=4, stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        dummy = torch.zeros(
            1, obs_space.shape[0], obs_space.shape[1], obs_space.shape[2]
        )
        out = self.conv(dummy)
        conv_out_size = out.view(out.size(0), -1).size(1)

        # For single action spaces
        act_space_size = (
            act_space.nvec.prod()
            if isinstance(act_space, spaces.MultiDiscrete)
            else act_space.n
        )

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
        # Normalize each layer separately
        x[:, 0] = x[:, 0].float() / self.obs_space.high[0][0][0]  # Ally pos layer
        x[:, 1] = x[:, 1].float() / self.obs_space.high[1][0][0]  # Enemy pos layer
        x[:, 2] = x[:, 2].float()  # Cover pos layer (0-1)
        x[:, 3] = x[:, 3].float()  # Aiming layer (0-1)
        x[:, 4] = x[:, 4].float()  # Status layer (0-1)
        x[:, 5] = x[:, 5].float() / self.obs_space.high[5][0][0]  # Danger layer (0-100)

        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

    def save(self, filepath: str) -> None:
        directory = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath))
