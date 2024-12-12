from utils.server import server
from utils.logger import logger
from agent.state import StateCollector

from random import randint

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class PawnAction:
    label: str  # Fixed typo from 'Lable' to 'label'
    x: int
    y: int


@dataclass
class GameAction:
    pawn_actions: Dict[str, PawnAction]


# From another class/function:
def random_action_test():

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
    # Convert GameAction to dictionary before sending
    message = {"Type": "GameAction", "Data": asdict(game_action)}
    for client in server.clients:
        server.send_to_client(client, message)

    logger.debug(f"Sent random actions to clients\n")
