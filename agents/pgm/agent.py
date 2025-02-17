import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributions as distributions
import torch.optim as optim
from gymnasium.spaces import Box
from numpy.typing import NDArray

from env.utils import index_to_coord_batch
from .memory import PGMemory
from .model import PolicyNetwork


class PGAgent:
    def __init__(
        self,
        n_envs,
        obs_space: Box,
        act_space: Box,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.n_envs = n_envs
        self.act_space = act_space
        self.device = device

        self.gamma = 0.975
        self.entropy_coef_start = 0.005
        self.min_entropy_coef = self.entropy_coef_start
        self.entropy_decay = 0.99999
        self.entropy_coef = self.entropy_coef_start

        self.steps = 0

        self.policy_loss_history = []
        self.loss_history = []
        self.n_returns_history = []
        self.entropy_histroy = []
        self.entropy_coef_history = []
        self.entropy_bonus_history = []

        self.policy = PolicyNetwork(obs_space, act_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0005)
        self.memory = PGMemory()

    def act(self, states: NDArray):
        states_tensor_batch = torch.FloatTensor(states).to(self.device)
        self.steps += self.n_envs

        # Update entropy coefficient
        self.entropy_coef = max(
            self.entropy_coef_start * (1 - np.exp(-5 * self.entropy_decay**self.steps)),
            self.min_entropy_coef,
        )

        logits_batch = self.policy.forward(states_tensor_batch)
        dists_batch = distributions.Categorical(logits=logits_batch)
        raw_action_batch = dists_batch.sample()
        log_prob_batch = dists_batch.log_prob(raw_action_batch)
        batch_actions = (
            index_to_coord_batch(self.act_space, raw_action_batch.unsqueeze(-1))
            .cpu()
            .numpy()
            .astype(self.act_space.dtype)
        )

        return batch_actions, log_prob_batch

    def remember(
        self,
        state: NDArray,
        action: NDArray,
        log_prob: torch.Tensor,
        reward: float,
        next_state: NDArray,
        done: bool,
    ) -> None:
        self.memory.store(
            state=torch.tensor(state).to(self.device),
            action=torch.tensor(action).to(self.device),
            log_prob=log_prob,
            reward=torch.tensor(reward).to(self.device),
            next_state=torch.tensor(next_state).to(self.device),
            done=torch.tensor(done).to(self.device),
        )

    def train(self) -> None:
        returns = []
        G = 0
        for transition in self.memory.transitions:
            reward, done = transition.reward.to(self.device), transition.done.to(
                self.device
            )
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate policy loss
        log_probs = torch.stack([t.log_prob for t in self.memory.transitions]).to(
            self.device
        )
        policy_loss = -torch.sum(log_probs * returns) / len(returns)

        # Calculate entropy (optional for exploration)
        states = torch.stack([t.state for t in self.memory.transitions]).to(self.device)
        logits = self.policy.forward(states)  # Get logits for all actions
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        # Compute entropy
        entropy_bonus = self.entropy_coef * entropy

        # Total loss
        loss = policy_loss - entropy_bonus

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging
        self.policy_loss_history.append(policy_loss.item())
        self.entropy_histroy.append(entropy.item())
        self.loss_history.append(loss.item())
        self.n_returns_history.append(returns.mean().item())
        self.entropy_coef_history.append(self.entropy_coef)
        self.entropy_bonus_history.append(entropy_bonus.item())

        self.memory.clear()

    def draw(self, save_path: str = "./training_history.png") -> None:
        """
        Plots the training statistics including entropy coefficient decay over the training steps.

        Args:
            save_path (str, optional): Path to save the plot. Defaults to "./training_history.png".
        """
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create a DataFrame with the training statistics
        stats_df = pd.DataFrame(
            {
                "Update": range(len(self.loss_history)),
                "Policy Loss": self.policy_loss_history,
                "Total Loss": self.loss_history,
                "Norm Return": self.n_returns_history,
                "Entropy": self.entropy_histroy,
                "Entropy Coefficient": self.entropy_coef_history,  # Added entropy coefficient
                "Entropy Bonus": self.entropy_bonus_history,  # Added entropy bonus
            }
        )

        # Create subplots for each metric
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
            6, 1, figsize=(10, 22)
        )  # Added fifth subplot

        # Plot Policy Loss
        sns.lineplot(data=stats_df, x="Update", y="Policy Loss", ax=ax1)
        ax1.set_title("Policy Loss over Updates")

        # Plot Total Loss
        sns.lineplot(data=stats_df, x="Update", y="Total Loss", ax=ax2)
        ax2.set_title("Total Loss over Updates")

        # Plot Return History
        sns.lineplot(data=stats_df, x="Update", y="Norm Return", ax=ax3)
        ax3.set_title("Norm Return over Updates")

        # Plot Entropy
        sns.lineplot(data=stats_df, x="Update", y="Entropy", ax=ax4)
        ax4.set_title("Entropy over Updates")

        # Plot Entropy Coefficient Decay
        sns.lineplot(data=stats_df, x="Update", y="Entropy Coefficient", ax=ax5)
        ax5.set_title("Entropy Coefficient Decay over Updates")

        # Plot Entropy Bonus
        sns.lineplot(data=stats_df, x="Update", y="Entropy Bonus", ax=ax6)
        ax6.set_title("Entropy Bonus over Updates")

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
