import torch
import torch.optim as optim
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
        lr: float = 1e-4,
        gamma: float = 0.99,
        k_epochs: int = 6,
        eps_clip: float = 0.1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        batch_size: int = 128,
        reuse_time: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.batch_size = batch_size
        self.reuse_time = reuse_time
        self.state_values_store = []

        self.policy = ActorCritic(obs_space, act_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = PPOMemory()

    def select_action(self, state: torch.Tensor) -> Dict:
        state = torch.FloatTensor(state).to(self.device)
        action, log_prob, state_values = self.policy.act(state)
        self.state_values_store.append(state_values)
        self.current_transition = {
            "state": state,
            "action": torch.tensor([action[0][0], action[1][0]]),
            "log_prob": log_prob,
        }
        action = {1: np.array(action)}
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
            log_probs, entropy, state_values = self.policy.evaluate(batch_states)

            ratios = torch.exp(log_probs - batch_old_log_probs.detach())
            # surr1 = ratios * batch_advantages
            surr1 = log_probs.detach() * batch_advantages.detach()
            # surr2 = (
            #     torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
            #     * batch_advantages
            # )
            entropy_bonus = self.entropy_coef * entropy
            # actor_loss = -torch.min(surr1, surr2) - entropy_bonus
            actor_loss = -surr1.mean() - entropy_bonus.mean()
            # critic_loss = (
            #     self.critic_coef * 0.5 * (batch_returns - state_values).pow(2)
            # )
            critic_loss = self.critic_coef * batch_advantages.detach().mean().pow(2)
            # critic_loss = batch_advantages.pow(2)

            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.state_values_store = []
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

        advantages = torch.tensor(
            advantages, dtype=torch.float32, requires_grad=True
        ).to(self.device)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages
