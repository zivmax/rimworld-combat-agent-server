import torch
import torch.optim as optim
from typing import Dict
from gymnasium.spaces import Box
import numpy as np
from numpy.typing import NDArray
from torch import distributions
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from env.utils import index_to_coord_batch
from .memory import PPOMemory
from .model import ActorCritic


class PPOAgent:
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
        self.k_epochs = 8
        self.eps_clip = 0.1
        self.batch_size = 1024

        self.entropy_coef_start = 1.0
        self.min_entropy_coef = 0.005
        self.entropy_decay = 0.99999
        self.entropy_coef = self.entropy_coef_start

        self.critic_coef = 1.0

        self.state_values_buffer = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.loss_history = []
        self.entropy_history = []
        self.entropy_coef_history = []
        self.entropy_bonus_history = []
        self.advantages_history = []
        self.surr_history = []
        self.steps = 0

        self.policy = ActorCritic(obs_space, act_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1.5e-5)
        self.memory = PPOMemory()

    def act(self, states: NDArray) -> tuple[NDArray, torch.Tensor, torch.Tensor]:
        self.steps += self.n_envs
        with torch.no_grad():
            states_tensor_batch = torch.FloatTensor(np.array(states)).to(self.device)
            logits_batch, state_values_batch = self.policy.forward(states_tensor_batch)
            dists_batch = distributions.Categorical(logits=logits_batch)
            raw_action_batch = dists_batch.sample()
            log_prob_batch = dists_batch.log_prob(raw_action_batch)

            batch_actions = (
                index_to_coord_batch(self.act_space, raw_action_batch.unsqueeze(1))
                .cpu()
                .numpy()
                .astype(self.act_space.dtype)
            )

            # Store state values
            self.state_values_buffer.extend(state_values_batch.cpu().detach().numpy())

            return (
                batch_actions,
                log_prob_batch,
            )

    def remember(
        self,
        state: NDArray,
        action: NDArray,
        log_prob: torch.Tensor,
        reward: float,
        next_state: NDArray,
        done: bool,
    ) -> None:
        self.memory.store_transition(
            state=torch.tensor(state).to(self.device),
            action=torch.tensor(action).to(self.device),
            log_prob=log_prob.to(self.device),
            reward=torch.tensor(reward).to(self.device),
            next_state=torch.tensor(next_state).to(self.device),
            done=torch.tensor(done).to(self.device),
        )

    def train(self) -> None:
        for _ in range(self.k_epochs):
            states = torch.stack([t.state for t in self.memory.transitions])
            actions = torch.stack([t.action for t in self.memory.transitions])
            old_log_probs = torch.stack([t.log_prob for t in self.memory.transitions])
            rewards = torch.stack([t.reward for t in self.memory.transitions])
            dones = torch.stack([t.done for t in self.memory.transitions])

            returns, advantages = self.compute_advantages(rewards, dones)

            batch_size = min(self.batch_size, len(self.memory.transitions))

            dataset = torch.utils.data.TensorDataset(
                states, actions, old_log_probs, returns, advantages
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

            for batch in loader:
                (
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_returns,
                    batch_advantages,
                ) = batch

                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_old_log_probs = batch_old_log_probs.to(self.device)
                batch_returns = batch_returns.to(self.device)
                batch_advantages = batch_advantages.to(self.device)

                log_probs, entropy, state_values = self.policy.evaluate(batch_states)

                ratios = torch.exp(log_probs - batch_old_log_probs.detach())

                surr = ratios * batch_advantages
                surr_clamp = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * batch_advantages
                )

                entropy_bonus = self.entropy_coef * entropy

                actor_loss = -torch.min(surr, surr_clamp) - entropy_bonus

                critic_loss = (
                    self.critic_coef * 0.5 * torch.abs(batch_returns - state_values)
                )

                loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

                self.optimizer.step()

                self.policy_loss_history.append(actor_loss.mean().item())
                self.value_loss_history.append(critic_loss.mean().item())
                self.loss_history.append(loss.mean().item())
                self.entropy_history.append(entropy.mean().item())
                self.entropy_coef_history.append(self.entropy_coef)
                self.entropy_bonus_history.append(entropy_bonus.mean().item())
                self.advantages_history.append(batch_advantages.mean().item())
                self.surr_history.append(surr.mean().item())

        self.entropy_coef = max(
            self.entropy_coef_start * (1 - np.exp(-5 * self.entropy_decay**self.steps)),
            self.min_entropy_coef,
        )
        self.state_values_buffer.clear()
        self.memory.clear()

    def compute_advantages(self, rewards, dones):
        GAE_LAMBDA = 0.9
        GAMMA = self.gamma
        state_values = self.state_values_buffer
        advantages = []
        returns = []
        advantage = 0
        previous_value = 0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step].float().cpu().numpy()
            delta = (
                rewards[step].cpu().numpy()
                + GAMMA * previous_value * mask
                - state_values[step]
            )
            advantage = delta + GAMMA * GAE_LAMBDA * mask * advantage
            advantages.insert(0, advantage)
            previous_value = state_values[step]
            returns.insert(0, advantage + state_values[step])

        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(
            self.device
        )

        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns, advantages

    def draw(self, save_path: str = "./training_history.png") -> None:
        """
        Plots the training statistics for PPO.

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
                "Value Loss": self.value_loss_history,
                "Total Loss": self.loss_history,
                "Entropy": self.entropy_history,
                "Entropy Coef": self.entropy_coef_history,
                "Entropy Bonus": self.entropy_bonus_history,
                "Advantages": self.advantages_history,
                "Surrogate": self.surr_history,
            }
        )

        # Create subplots for each metric
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(
            8, 1, figsize=(10, 26)
        )

        # Plot Policy Loss
        sns.lineplot(data=stats_df, x="Update", y="Policy Loss", ax=ax1)
        ax1.set_title("Policy Loss over Updates")

        # Plot Value Loss
        sns.lineplot(data=stats_df, x="Update", y="Value Loss", ax=ax2)
        ax2.set_title("Value Loss over Updates")

        # Plot Total Loss
        sns.lineplot(data=stats_df, x="Update", y="Total Loss", ax=ax3)
        ax3.set_title("Total Loss over Updates")

        # Plot Entropy
        sns.lineplot(data=stats_df, x="Update", y="Entropy", ax=ax4)
        ax4.set_title("Entropy over Updates")

        # Plot Entropy Coef
        sns.lineplot(data=stats_df, x="Update", y="Entropy Coef", ax=ax5)
        ax5.set_title("Entropy Coef over Updates")

        # Plot Advantages
        sns.lineplot(data=stats_df, x="Update", y="Advantages", ax=ax6)
        ax6.set_title("Advantages over Updates")

        # Plot Surrogate
        sns.lineplot(data=stats_df, x="Update", y="Surrogate", ax=ax7)
        ax7.set_title("Surrogate over Updates")

        # Plot Entropy Bonus
        sns.lineplot(data=stats_df, x="Update", y="Entropy Bonus", ax=ax8)
        ax8.set_title("Entropy Bonus over Updates")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
