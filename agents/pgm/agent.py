import torch
import torch.optim as optim
from typing import Dict
from gymnasium.spaces import Box
import numpy as np
from numpy.typing import NDArray
from .memory import PGMemory, Transition  # Renamed if necessary
from .model import PolicyNetwork
import torch.distributions as distributions


class PGAgent:
    def __init__(
        self,
        n_envs,
        obs_space: Box,
        act_space: Box,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.n_envs = n_envs
        self.act_space = act_space
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.policy_loss_history = []
        self.loss_history = []

        self.policy = PolicyNetwork(obs_space, act_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = PGMemory()  # Adjusted memory class if necessary

    def act(self, states: NDArray) -> tuple[NDArray, NDArray]:
        states = np.array(states)
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_batch = np.zeros((self.n_envs, 2), dtype=self.act_space.dtype)
        log_probs_batch = np.zeros(self.n_envs, dtype=np.float32)

        for i in range(self.n_envs):
            action_mean, action_std = self.policy.forward(states_tensor[i])

            dist_x, dist_y = distributions.Normal(
                action_mean[0, 0], action_std[0, 0]
            ), distributions.Normal(action_mean[0, 1], action_std[0, 1])

            action_x, action_y = dist_x.sample(), dist_y.sample()

            log_probs = dist_x.log_prob(action_x) + dist_y.log_prob(action_y)
            actions = [action_x.cpu().item(), action_y.cpu().item()]

            actions[0] = max(
                min(actions[0], self.act_space.high[0]),
                self.act_space.low[0],
            )
            actions[1] = max(
                min(actions[1], self.act_space.high[1]),
                self.act_space.low[1],
            )

            actions_batch[i] = actions
            log_probs_batch[i] = log_probs.cpu().item()

        return actions_batch, log_probs_batch

    def remember(
        self,
        state: NDArray,
        action: NDArray,
        log_prob: NDArray,
        reward: float,
        next_state: NDArray,
        done: bool,
    ) -> None:
        self.memory.store(
            state=torch.tensor(state, device="cpu"),
            action=torch.tensor(action, device="cpu"),
            log_prob=torch.tensor(log_prob, device="cpu"),
            reward=torch.tensor(reward, device="cpu"),
            next_state=torch.tensor(next_state, device="cpu"),
            done=torch.tensor(done, device="cpu"),
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
        entropy = torch.stack(
            [log_prob.exp() * log_prob for log_prob in log_probs]
        ).mean()
        entropy_loss = -self.entropy_coef * entropy

        # Total loss
        loss = policy_loss + entropy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging
        self.policy_loss_history.append(policy_loss.item())
        self.loss_history.append(loss.item())

        # Clear memory
        self.memory.clear()
