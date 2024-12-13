from agent import Agent 
from agent import GameAction, PawnAction

from utils.logger import logger
from utils.json import to_json

from random import randint

class RandomAgent(Agent):

    def __init__(self):
        super().__init__()

    def act(self) -> GameAction:

        # From GameState to fetch the labels of the ally pawn
        ally_pawn_labels = []
        for pawn in self.state.pawn_states.values():
            if pawn.is_ally:
                ally_pawn_labels.append(pawn.label)

        # For each ally pawn, create a random action
        pawn_actions = {}
        for label in ally_pawn_labels:
            pawn_actions[label] = PawnAction(
                label=label,
                x=randint(0, self.state.map_state.width),
                y=randint(0, self.state.map_state.height),
            )

        logger.debug(
            f"Random actions (tick {self.state.tick}): \n{to_json(pawn_actions, indent=2)}\n"
        )

        return GameAction(pawn_actions)


    def save(self):
        pass

    def load(self):
        pass