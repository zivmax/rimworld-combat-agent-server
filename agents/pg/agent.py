import torch
import torch.optim as optim
from typing import Dict
from gymnasium.spaces import Box
import numpy as np
from numpy.typing import NDArray
from .memory import PGMemory, Transition  # Renamed if necessary
from .model import PolicyNetwork


class PGAgent:
    def __init__(
        self,
        n_envs,
        obs_space: Box,
        act_space: Box,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        batch_size: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.n_envs = n_envs
        self.act_space = act_space
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.current_transitions = []
        self.policy_loss_history = []
        self.loss_history = []

        self.policy = PolicyNetwork(obs_space, act_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = PGMemory()  # Adjusted memory class if necessary

    def select_action(self, states: NDArray) -> NDArray:
        with torch.no_grad():
            states = np.array(states)
            states_tensor = torch.FloatTensor(states).to(self.device)
            batch_actions = np.zeros((self.n_envs, 2), dtype=self.act_space.dtype)

            for i in range(self.n_envs):
                actions, log_probs = self.policy.act(states_tensor[i])
                actions[0] = max(
                    min(actions[0], self.act_space.high[0]),
                    self.act_space.low[0],
                )
                actions[1] = max(
                    min(actions[1], self.act_space.high[1]),
                    self.act_space.low[1],
                )

                batch_actions[i] = actions

                self.current_transitions.append(
                    {
                        "state": states_tensor[i],
                        "action": torch.tensor(actions).to(self.device),
                        "log_prob": log_probs,
                    }
                )
        return batch_actions

    def store_transition(
        self,
        reward: float,
        next_state: NDArray,
        done: bool,
    ) -> None:
        for transition in self.current_transitions:
            self.memory.store_transition(
                state=transition["state"],
                action=transition["action"],
                log_prob=transition["log_prob"],
                reward=torch.tensor(reward).to(self.device),
                next_state=torch.tensor(next_state).to(self.device),
                done=torch.tensor(done).to(self.device),
            )
        self.current_transitions = []

    def update(self) -> None:

        returns = []
        G = 0
        for reward, done in zip(self.memory.rewards, self.memory.dones):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate policy loss
        log_probs = torch.stack(self.memory.log_probs)
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
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Logging
        self.policy_loss_history.append(policy_loss.item())
        self.loss_history.append(loss.item())

        # Clear memory
        self.memory.clear()
