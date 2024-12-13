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
    @dataclass
    class CombatStats:
        melee_DPS: float
        shooting_ACC: float
        move_speed: float

        @classmethod
        def from_dict(cls, data: Dict[str, float]) -> "PawnState.CombatStats":
            return cls(
                melee_DPS=data["MeleeDPS"],
                shooting_ACC=data["ShootingACC"],
                move_speed=data["MoveSpeed"],
            )

    @dataclass
    class HealthStats:
        pain_shock: float
        blood_loss: float

        @classmethod
        def from_dict(cls, data: Dict[str, float]) -> "PawnState.HealthStats":
            return cls(pain_shock=data["PainShock"], blood_loss=data["BloodLoss"])

    @dataclass
    class Loc:
        x: int
        y: int

        @classmethod
        def from_dict(cls, data: Dict[str, int]) -> "PawnState.Loc":
            return cls(x=data["X"], y=data["Y"])

    label: str
    is_ally: bool
    loc: Loc
    equipment: str
    combat_stats: CombatStats
    health_stats: HealthStats
    is_incapable: bool

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "PawnState":
        return cls(
            label=data["Label"],
            is_ally=data["IsAlly"],
            loc=cls.Loc.from_dict(data["Loc"]),
            equipment=data["Equipment"],
            combat_stats=cls.CombatStats.from_dict(data["CombatStats"]),
            health_stats=cls.HealthStats.from_dict(data["HealthStats"]),
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
