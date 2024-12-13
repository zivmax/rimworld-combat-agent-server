from utils.server import server
from utils.logger import logger
from utils.json import to_json
from agent.state import StateCollector

from random import randint
from dataclasses import dataclass
from typing import Dict


@dataclass
class PawnAction:
    label: str
    x: int
    y: int

    def __iter__(self):
        yield ("Label", self.label)
        yield ("X", self.x)
        yield ("Y", self.y)


@dataclass
class GameAction:
    pawn_actions: Dict[str, PawnAction]

    def __iter__(self):
        yield ("PawnActions", {k: dict(v) for k, v in self.pawn_actions.items()})


def random_action_test() -> None:

    state = StateCollector.current_state

    # From GameState to fetch the labels of the ally pawn
    ally_pawn_labels = []
    for label, pawn in state.pawn_states.items():
        if pawn.is_ally:
            ally_pawn_labels.append(label)

    # For each ally pawn, create a random action
    pawn_actions = {}
    for label in ally_pawn_labels:
        pawn_actions[label] = PawnAction(
            label=label,
            x=randint(0, state.map_state.width),
            y=randint(0, state.map_state.height),
        )

    game_action = GameAction(pawn_actions)

    message = {
        "Type": "GameAction",
        "Data": dict(game_action),
    }
    for client in server.clients:
        server.send_to_client(client, message)

    logger.info(f"Sent random actions to clients\n")
    logger.debug(f"Random actions: \n{to_json(game_action, indent=2)}")
