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
        self.act_space = act_space
        self.act_size = int(np.prod(self.act_space.high - self.act_space.low + 1))
        self.dim_actions = self.act_size  # Total number of possible actions (9)

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

        # Actor outputs logits for each possible action index (0 to 8)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.dim_actions),  # Output logits for each action index
        )

        # Critic remains unchanged
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.eval_critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, eval: bool = False):
        x = state.float()
        x = x.unsqueeze(0) if len(x.shape) != 5 else x
        x = x / (torch.tensor(self.obs_space.high, device=x.device)).repeat(
            x.shape[0], *(len(self.obs_space.high.shape) * [1])
        )
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        action_logits = self.actor(x)  # Logits for each action index
        state_values = self.critic(x) if not eval else self.eval_critic(x)
        return action_logits, state_values

    def evaluate(self, states: torch.Tensor):
        action_logits, state_values = self.forward(states, eval=True)
        dist = distributions.Categorical(
            logits=action_logits
        )  # Categorical distribution
        action = dist.sample()  # Sample action index
        action_log_prob = dist.log_prob(action)  # Log probability of the sampled action
        entropy = dist.entropy()  # Entropy of the distribution
        return action_log_prob, entropy, state_values

    def save(self, filepath: str) -> None:
        directory = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath))
        self.eval()
