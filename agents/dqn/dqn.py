
import numpy as np
import torch
from typing import Tuple, Dict
from utils.logger import logger

from agents.agent import Agent
from env.action import GameAction
from env.state import PawnState
from gymnasium.spaces import MultiDiscrete
from env.rimworld import RimWorldEnv
from .network import DQNModel
from hyper_params import N_EPISODES, EPISOLD_LOG_INTERVAL, EPISOLD_SAVE_INTERVAL


class DQNAgent(Agent):
    def __init__(
        self,
        state_dim: Tuple[int, int],
        action_space: MultiDiscrete
    ):
        """
        Initialize the DQNAgent.

        Args:
            state_dim (Tuple[int, int]): Dimension of the state space (height, width).
            action_space (MultiDiscrete): Action space.
            ... [Other Args]
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_space = action_space


        self.action_dim = self._get_action_dim()

        self.model = DQNModel(
            state_size=self._get_state_size(),
            action_size=self.action_dim,
        )

    def _get_state_size(self) -> int:
        """Input size."""
        return self.state_dim[0] * self.state_dim[1]

    def _get_action_dim(self) -> int:
        """Calculate the total number of possible actions."""
        action_dim = 1
        for space in self.action_space.spaces.values():
            action_dim *= space.nvec.prod()
        return action_dim

    def _index_to_action(self, index: int) -> Dict[int, Tuple[int, int]]:
        """
        Convert a flat action index back to a structured action per ally.

        Args:
            index (int): Flat action index.

        Returns:
            Dict[int, Tuple[int, int]]: Structured action dictionary.
        """
        action = {}
        for ally_id, space in self.action_space.spaces.items():
            n = space.nvec.prod()
            action_part = index % n
            action[ally_id] = (
                action_part // space.nvec[1],
                action_part % space.nvec[1],
            )
            index = index // n
        return action

    def _action_to_index(self, action: GameAction) -> int:
        """
        Convert a structured GameAction to a flat action index.

        Args:
            action (GameAction): Structured action.

        Returns:
            int: Flat action index.
        """
        index = 0
        for ally_id, (x, y) in action.action_dict.items():
            space = self.action_space.spaces[ally_id]
            n = space.nvec.prod()
            part = x * space.nvec[1] + y
            index = index * n + part
        return index

    def act(self, obs: np.ndarray, info: Dict[int, PawnState]) -> GameAction:
        """
        Select an action based on the current observation.

        Args:
            obs (np.ndarray): Current state observation.
            info (Dict[int, PawnState]): Additional information about pawns.

        Returns:
            GameAction: Selected action encapsulated in GameAction.
        """
        state = obs.flatten()
        action_idx = self.model.act(state)
        action = self._index_to_action(action_idx)
        return GameAction(action)

    def step(self, state, action, reward, next_state, done):
        """
        Save experience to replay buffer and train the network.

        Args:
            state (np.ndarray): Current state.
            action (GameAction): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode is done.
        """
        # Flatten states to match the network input
        state = state.flatten()
        next_state = next_state.flatten()

        # Convert structured action to flat index
        action_idx = self._action_to_index(action)

        # Store experience and perform replay
        self.model.remember(state, action_idx, reward, next_state, done)
        self.model.replay()

        # update
        if self.model.steps_done % self.model.target_update == 0:
            self.model.update_target_network()

    def save(self, path: str):
        """
        Save the policy network parameters.

        Args:
            path (str): Path to save the model.
        """
        self.model.save(path)

    def load(self, path: str):
        """
        Load the policy network parameters.

        Args:
            path (str): Path to load the model from.
        """
        self.model.load(path)


def main():
    """
    Main function to train the DQNAgent.
    """
    env = RimWorldEnv()
    state_dim = (env.observation_space.shape[0], env.observation_space.shape[1])
    action_space = env.action_space

    agent = DQNAgent(state_dim=state_dim, action_space=action_space)

    num_episodes = N_EPISODES
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.step(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            if (episode + 1) % EPISOLD_LOG_INTERVAL == 0:
                logger.info(f"for episode {episode + 1}, reward: {reward}")

        logger.info(F"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        if (episode + 1) % EPISOLD_SAVE_INTERVAL == 0:
            agent.save(f"dqn_model_episode_{episode + 1}.pth")

    env.close()


if __name__ == "__main__":
    main()