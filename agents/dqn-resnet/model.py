import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from gymnasium import spaces
import numpy as np
import math
import torchvision.models.video as models


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Box,
        device="cpu",
        noisy=True,
        atoms=51,
        v_min=-10,
        v_max=10,
    ):
        super(DQN, self).__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.supports = torch.linspace(self.v_min, self.v_max, self.atoms)
        self.act_space_size = int(np.prod(act_space.high - act_space.low + 1))

        self.conv = models.r3d_18()

        def replace_bn_with_gn(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm3d):
                    setattr(
                        module,
                        name,
                        nn.GroupNorm(num_groups=32, num_channels=child.num_features),
                    )
                else:
                    replace_bn_with_gn(child)

        replace_bn_with_gn(self.conv)

        self.conv.stem = nn.Conv3d(
            in_channels=obs_space.shape[0],
            out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        self.conv.fc = nn.Identity()

        dummy = torch.zeros(1, *obs_space.shape)
        out = self.conv(dummy)
        conv_out_size = out.view(out.size(0), -1).size(1)

        # Dueling networks architecture
        LinearLayer = NoisyLinear if noisy else nn.Linear

        self.advantage_hidden = nn.Sequential(
            LinearLayer(conv_out_size, 1024),
            nn.ReLU(),
            LinearLayer(1024, 1024),
            nn.ReLU(),
            LinearLayer(1024, 1024),
            nn.ReLU(),
        )

        self.value_hidden = nn.Sequential(
            LinearLayer(conv_out_size, 1024),
            nn.ReLU(),
            LinearLayer(1024, 1024),
            nn.ReLU(),
            LinearLayer(1024, 1024),
            nn.ReLU(),
        )

        self.advantage = LinearLayer(1024, self.act_space_size * self.atoms)
        self.value = LinearLayer(1024, self.atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda if self.device == "cuda" else True
        assert x.is_cpu if self.device == "cpu" else True

        # Convert the entire tensor to float
        x = x.float()
        x = x / (
            torch.tensor(self.obs_space.high - self.obs_space.low, device=x.device)
        ).repeat(x.shape[0], *(len(self.obs_space.shape) * [1]))

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        advantage = self.advantage_hidden(x)
        value = self.value_hidden(x)

        advantage = self.advantage(advantage).view(-1, self.act_space_size, self.atoms)
        value = self.value(value).view(-1, 1, self.atoms)

        # Combine value and advantage using dueling formula
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_dist

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def save(self, filepath: str) -> None:
        directory = os.path.dirname(filepath) if os.path.dirname(filepath) else "."
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(
            torch.load(filepath, map_location=self.device, weights_only=True)
        )
