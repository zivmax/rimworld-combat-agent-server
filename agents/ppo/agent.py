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
        self.k_epochs = 5
        self.eps_clip = 0.1
        self.entropy_coef = 0.05
        self.critic_coef = 1.0
        self.batch_size = 128
        self.reuse_time = 8
        self.state_values_store = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.loss_history = []
        self.entropy_history = []
        self.advantages_history = []

        self.policy = ActorCritic(obs_space, act_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.00015)
        self.memory = PPOMemory()

    def act(self, states: NDArray) -> tuple[NDArray, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            states = np.array(states)
            states_tensor = torch.FloatTensor(states).to(self.device)
            batch_actions = np.zeros((self.n_envs, 2), dtype=self.act_space.dtype)
            batch_log_probs = []
            batch_state_values = []

            for i in range(self.n_envs):
                # Get action distributions and state values from policy network
                action_mean, action_std, state_values = self.policy.forward(
                    states_tensor[i]
                )

                # Create normal distributions for x and y coordinates
                dist_x = distributions.Normal(action_mean[0, 0], action_std[0, 0])
                dist_y = distributions.Normal(action_mean[0, 1], action_std[0, 1])

                # Sample actions
                action_x = dist_x.sample()
                action_y = dist_y.sample()

                # Calculate log probabilities
                action_log_prob = dist_x.log_prob(action_x) + dist_y.log_prob(action_y)

                # Convert to list and clip to action space bounds
                actions = [action_x.cpu().item(), action_y.cpu().item()]
                actions[0] = int(
                    round(
                        max(
                            min(actions[0], self.act_space.high[0]),
                            self.act_space.low[0],
                        )
                    )
                )
                actions[1] = int(
                    round(
                        max(
                            min(actions[1], self.act_space.high[1]),
                            self.act_space.low[1],
                        )
                    )
                )

                batch_actions[i] = actions
                batch_log_probs.append(action_log_prob)
                batch_state_values.append(state_values)

            self.state_values_store.extend(
                [v.cpu().detach().numpy() for v in batch_state_values]
            )
            return (
                batch_actions,
                torch.stack(batch_log_probs),
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
            state=torch.tensor(state).to("cpu"),
            action=torch.tensor(action).to("cpu"),
            log_prob=log_prob.to("cpu"),
            reward=torch.tensor(reward).to("cpu"),
            next_state=torch.tensor(next_state).to("cpu"),
            done=torch.tensor(done).to("cpu"),
        )

    def train(self) -> None:
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

            surr1 = ratios * batch_advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                * batch_advantages
            )

            entropy_bonus = self.entropy_coef * entropy

            actor_loss = -torch.min(surr1, surr2) - entropy_bonus

            critic_loss = self.critic_coef * 0.5 * (batch_returns - state_values).pow(2)

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            self.optimizer.step()

            self.policy_loss_history.append(actor_loss.mean().item())
            self.value_loss_history.append(critic_loss.mean().item())
            self.loss_history.append(loss.mean().item())
            self.entropy_history.append(entropy.mean().item())
            self.advantages_history.append(batch_advantages.mean().item())

        self.state_values_store.clear()
        self.memory.clear()

    def compute_advantages(self, rewards, dones):
        GAE_LAMBDA = 0.95
        GAMMA = self.gamma
        state_values = self.state_values_store
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
        return returns, advantages

    def draw(self, save_path: str = "./training_history.png") -> None:
        """
        Plots the training statistics for PPO (Policy Loss, Value Loss, Total Loss, Entropy, Advantages).

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
                "Advantages": self.advantages_history,
            }
        )

        # Create subplots for each metric
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 20))

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

        # Plot Advantages
        sns.lineplot(data=stats_df, x="Update", y="Advantages", ax=ax5)
        ax5.set_title("Advantages over Updates")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
