import torch
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict
from gymnasium.spaces import Box
import numpy as np
from .memory import PPOMemory, Transition
from .model import ActorCritic


class PPOAgent:
    def __init__(
        self,
        obs_space: Box,
        act_space: Box,
        lr: float = 3e-4,
        gamma: float = 0.99,
        k_epochs: int = 4,
        eps_clip: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.state_values_store = []

        # Updated to pass the entire act_space instead of act_space[1]
        self.policy = ActorCritic(obs_space, act_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = PPOMemory()

    def select_action(self, state: torch.Tensor) -> Dict:
        state = torch.FloatTensor(state).to(self.device)

        action, log_prob, state_values = self.policy.act(state)
        self.state_values_store.append(state_values)
        # Temporarily store partial transition information
        self.current_transition = {
            "state": state,
            "action": action,
            "log_prob": log_prob,
        }
        action = action.cpu().numpy()
        action = {1: np.array(action[0])}
        return action

    def store_transition(
        self,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        transition = self.current_transition
        self.memory.store_transition(
            state=transition["state"],
            action=transition["action"],
            log_prob=transition["log_prob"],
            reward=torch.tensor(reward).to(self.device),
            next_state=torch.tensor(next_state).to(self.device),
            done=torch.tensor(done).to(self.device),
        )
        self.current_transition = {}

    def update(self) -> None:

        states = torch.stack([t.state for t in self.memory.transitions]).to(self.device)
        actions = torch.stack([t.action for t in self.memory.transitions]).to(
            self.device
        )
        old_log_probs = torch.stack([t.log_prob for t in self.memory.transitions]).to(
            self.device
        )
        rewards = torch.stack([t.reward for t in self.memory.transitions]).to(
            self.device
        )
        dones = torch.stack([t.done for t in self.memory.transitions]).to(self.device)

        # Compute advantages and returns
        returns, advantages = self.compute_advantages(rewards, dones)

        for _ in range(self.k_epochs):

            log_probs, entropy, state_values = self.policy.evaluate(states, actions)

            ratios = torch.exp(log_probs - old_log_probs.detach())

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = -torch.min(surr1, surr2) + 0.5 * (returns - state_values).pow(2)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

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
            mask = 1.0 - dones[step].float()
            delta = rewards[step] + GAMMA * previous_value * mask - state_values[step]
            advantage = delta + GAMMA * GAE_LAMBDA * mask * advantage
            advantages.insert(0, advantage)
            previous_value = state_values[step]
            returns.insert(0, advantage + state_values[step])
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages
