from agent.state import StateCollector, GameState, GameStatus, Loc, MapState, PawnState
from agent.action import GameAction, PawnAction


from abc import ABC, abstractmethod


class Agent(ABC):
    """Abstract base class for reinforcement learning agents."""


    def __init__(self):
        """
        Initialize the agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        self.state = StateCollector.current_state
        self.save_path = None

    @abstractmethod
    def act(self) -> GameAction:
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