from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Dict, Tuple
from gymnasium.spaces import MultiDiscrete

from env.action import GameAction
from env.state import PawnState, MapState, Loc


class Agent(ABC):
    """Abstract base class for reinforcement learning agents."""

    def __init__(self, act_space: Dict[int, MultiDiscrete], obs_space: MultiDiscrete):
        """
        Initialize the agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        self.act_space = act_space
        self.obs_space = obs_space

    @abstractmethod
    def act(
        self,
        obs: NDArray,
        info: Dict[
            str,
            MapState
            | Dict[int, PawnState]
            | Dict[int, MultiDiscrete]
            | Dict[int, tuple[Loc]],
        ],
    ) -> NDArray:
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
