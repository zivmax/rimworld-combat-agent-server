from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Dict, Tuple
from gymnasium.spaces import MultiDiscrete

from env.action import GameAction
from env.state import PawnState, MapState


class Agent(ABC):
    """Abstract base class for reinforcement learning agents."""

    def __init__(self, action_space: Dict, observation_space: MultiDiscrete):
        """
        Initialize the agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        self.action_space = action_space
        self.observation_space = observation_space

    @abstractmethod
    def act(
        self,
        obs: NDArray,
        info: Tuple[
            MapState,
            Dict[int, PawnState],
            Dict[int, Dict[str, Tuple[NDArray] | MultiDiscrete]],
        ],
    ) -> GameAction:
        """
        Select an action based on the current state.

        Args:
            state: Current state observation

        Returns:
            action: Selected action
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save the agent's model parameters.

        Args:
            path: Path to save the model
        """
        pass

    @abstractmethod
    def load(self):
        """
        Load the agent's model parameters.

        Args:
            path: Path to load the model from
        """
        pass
