import numpy as np
from typing import Tuple, Dict
from utils.logger import logger

from agents import Agent
from env.action import GameAction, PawnAction
from env.state import PawnState
from gymnasium.spaces import MultiDiscrete
from env import rimworld_env
from .network import DQNModel


class DQNAgent(Agent):
    def __init__(
        self,
        observation_space: MultiDiscrete,
        action_space: Dict[int, MultiDiscrete],
    ):
        """
        Initialize the DQNAgent.

        Args:
            state_dim (Tuple[int, int]): Dimension of the state space (height, width).
            action_space (MultiDiscrete): Action space.
            ... [Other Args]
        """
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.state_dim = observation_space.shape
        self.action_dim = self._get_action_dim()

        logger.info(
            f"Initializing DQNModel with state_size={self._get_state_size()}, action_size={self.action_dim}"
        )
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
        for space in self.action_space.values():
            action_dim *= space.nvec.prod()
        logger.info(f"Computed action_dim: {action_dim}")
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
        for ally_id, space in self.action_space.items():
            n = space.nvec.prod()
            action_part = index % n
            action[self.pawns[ally_id].label] = PawnAction(
                label=self.pawns[ally_id].label,
                x=int(action_part // space.nvec[1]),
                y=int(action_part % space.nvec[1]),
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
        for _, action in action.pawn_actions.items():
            label, x, y = action
            ally_id = None
            for idx, pawn in dict(self.pawns).items():
                if pawn.label == label[1]:
                    ally_id = idx

            if ally_id is None:
                raise ValueError(f"No pawn found with label {label}")

            # Access action space directly from the Dict
            space = self.action_space[ally_id]
            n = space.nvec.prod()
            part = x[1] * space.nvec[1] + y[1]
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
        self.pawns = info["pawns"]
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
