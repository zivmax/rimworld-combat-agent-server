from utils.server import server
from utils.logger import logger
from agent.state import StateCollector

from random import randint

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PawnAction:
    label: str = field(metadata={"json_name": "Label"})
    x: int = field(metadata={"json_name": "X"})
    y: int = field(metadata={"json_name": "Y"})


@dataclass
class GameAction:
    pawn_actions: Dict[str, PawnAction] = field(metadata={"json_name": "PawnActions"})


def convert_to_json_format(obj):
    if hasattr(obj, "__dict__"):
        result = {}
        for key, value in obj.__dict__.items():
            field_info = obj.__dataclass_fields__[key]
            json_key = field_info.metadata.get("json_name", key)
            if isinstance(value, dict):
                value = {k: convert_to_json_format(v) for k, v in value.items()}
            elif hasattr(value, "__dict__"):
                value = convert_to_json_format(value)
            result[json_key] = value
        return result
    return obj


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
    message = {
        "Type": "GameAction",
        "Data": convert_to_json_format(game_action),
    }
    for client in server.clients:
        server.send_to_client(client, message)

    logger.debug(f"Sent random actions to clients\n")
