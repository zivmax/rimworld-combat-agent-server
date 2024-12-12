from utils.server import server
from utils.logger import logger

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MapState:
    width: int
    height: int
    cells: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "MapState":
        return cls(width=data["Width"], height=data["Height"], cells=data["Cells"])


@dataclass
class PawnState:
    label: str
    is_ally: bool
    loc: Dict[str, int]
    equipment: str
    combat_stats: Dict[str, float]
    health_stats: Dict[str, float]
    is_incapable: bool

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "PawnState":
        return cls(
            label=data["Label"],
            is_ally=data["IsAlly"],
            loc=data["Loc"],
            equipment=data["Equipment"],
            combat_stats=data["CombatStats"],
            health_stats=data["HealthStats"],
            is_incapable=data["IsIncapable"],
        )


@dataclass
class GameState:
    map_state: MapState
    pawn_states: Dict[str, PawnState]
    tick: int
    game_ending: bool

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "GameState":

        # Create MapState
        map_state = MapState.from_dict(data["MapState"])

        # Create PawnStates
        pawn_states = {}
        for pawn_label, pawn_data in data["PawnStates"].items():
            pawn_states[pawn_label] = PawnState.from_dict(pawn_data)

        return cls(
            map_state=map_state,
            pawn_states=pawn_states,
            tick=data["Tick"],
            game_ending=data["GameEnding"],
        )


class StateCollector:
    current_state = None

    @classmethod
    def collect_state(cls) -> None:
        while True:
            if server.message_queue.qsize() > 0:
                # Peek at message without removing it
                message = server.message_queue.queue[0]
                # Only get the message if it's a state message
                if "GameState" == message["Type"]:
                    message = server.message_queue.get()
                    cls.current_state = GameState.from_dict(message["Data"])
                    logger.debug(
                        f"Collected game state at tick {cls.current_state.tick}\n"
                    )
                    break
