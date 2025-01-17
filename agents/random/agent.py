from random import randint
from numpy.typing import NDArray
from typing import Dict, Tuple
from gymnasium.spaces import Box
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

    def __init__(self, act_space: Box, obs_space: Box, n_envs: int):
        self.act_space = act_space
        self.obs_space = obs_space
        self.n_envs = n_envs

    def act(
        self,
        obs: NDArray,
    ) -> GameAction:
        space = self.act_space

        acts = space.sample()

        return acts

    def save(self):
        pass

    def load(self):
        pass
