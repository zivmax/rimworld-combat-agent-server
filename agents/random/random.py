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
        obs: NDArray,
        info: Dict[
            str,
            MapState
            | Dict[int, PawnState]
            | Dict[int, MultiDiscrete]
            | Dict[int, tuple[Loc]],
        ],
    ) -> GameAction:
        space = self.action_space
        map = info["map"]
        pawns = info["pawns"]
        mask = info["action_mask"]

        allies = [p for p in pawns.values() if p.is_ally]

        # For each ally pawn, create a random action
        ally_actions = {}

        for idx, ally in enumerate(allies, start=1):
            while True:
                act = space[idx].sample()
                if Loc(act[0], act[1]) not in mask[idx]:
                    break
            
        return {1: np.array([act[0], act[1]])}

        #     ally_actions[ally.label] = PawnAction(
        #         label=ally.label,
        #         x=int(act[0]),
        #         y=int(act[1]),
        #     )

        # logger.debug(f"Random actions: \n{to_json(ally_actions, indent=2)}")

        # return GameAction(ally_actions)

    def save(self):
        pass

    def load(self):
        pass
