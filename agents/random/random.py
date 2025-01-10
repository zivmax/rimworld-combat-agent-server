from random import randint
from numpy.typing import NDArray
from typing import Dict, Tuple
from gymnasium.spaces import MultiDiscrete
import numpy as np
from utils.json import to_json
from env.state import PawnState, MapState, Loc
from env.action import GameAction, PawnAction
from agents import Agent
import logging
from utils.logger import get_file_logger, get_cli_logger
from utils.timestamp import timestamp

logging_level = logging.DEBUG
f_logger = get_file_logger(
    __name__, f"agents/random/logs/{timestamp}.log", logging_level
)
cli_logger = get_cli_logger(__name__, logging_level)

logger = f_logger


class RandomAgent(Agent):

    def __init__(self, action_space: Dict, observation_space: MultiDiscrete):
        super().__init__(action_space, observation_space)

    def act(
        self,
    ) -> GameAction:
        space = self.action_space

        act = space[1].sample()

        return {1: act}

    def save(self):
        pass

    def load(self):
        pass
