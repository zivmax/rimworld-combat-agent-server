import torch
import torch.optim as optim
from typing import Dict
from gymnasium.spaces import Box
import numpy as np
from numpy.typing import NDArray
from .memory import PPOMemory, Transition
from .model import ActorCritic


class PPOAgent:
    def __init__(
        self,
        n_envs,
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
        self.n_envs = n_envs
        self.act_space = act_space
        self.device = device
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.batch_size = batch_size
        self.reuse_time = reuse_time
        self.state_values_store = []
        self.current_transitions = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.loss_history = []

        self.policy = ActorCritic(obs_space, act_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = PPOMemory()

    def select_action(self, states: NDArray) -> NDArray:
        with torch.no_grad():
            states = np.array(states)
            states_tensor = torch.FloatTensor(states).to(self.device)
            batch_actions = np.zeros((self.n_envs, 2), dtype=self.act_space.dtype)

            for i in range(self.n_envs):
                actions, log_probs, state_values = self.policy.act(states_tensor[i])
                actions[0] = max(
                    min(actions[0], self.act_space.high[0]),
                    self.act_space.low[0],
                )
                actions[1] = max(
                    min(actions[1], self.act_space.high[1]),
                    self.act_space.low[1],
                )

                self.state_values_store.extend(state_values.cpu().detach().numpy())

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
