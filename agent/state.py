from utils.server import server
from utils.logger import logger
from utils.json import to_json

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Loc:
    x: int
    y: int

    def __iter__(self):
        yield ("x", self.x)
        yield ("y", self.y)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "Loc":
        return cls(x=data["X"], y=data["Y"])


@dataclass
class MapState:
    width: int
    height: int
    cells: List[List["MapState.CellState"]]

    @dataclass
    class CellState:
        loc: Loc
        is_wall: bool
        is_tree: bool
        is_pawn: bool

        def __iter__(self):
            yield ("loc", dict(self.loc))
            yield ("is_wall", self.is_wall)
            yield ("is_tree", self.is_tree)
            yield ("is_pawn", self.is_pawn)

        @classmethod
        def from_dict(cls, data: Dict[str, bool]) -> "MapState.CellState":
            return cls(
                loc=Loc.from_dict(data["Loc"]),
                is_wall=data["IsWall"],
                is_tree=data["IsTree"],
                is_pawn=data["IsPawn"],
            )

    def __iter__(self):
        yield ("width", self.width)
        yield ("height", self.height)
        yield (
            "cells",
            {
                f"({x},{y})": dict(self.cells[x][y])
                for x in range(self.width)
                for y in range(self.height)
                if self.cells[x][y] is not None
            },
        )

    @classmethod
    def from_dict(cls, data: Dict[str, dict | int]) -> "MapState":
        # Init a 2D list of CellStates according to the map's width and height
        cells = []
        for _ in range(data["Width"]):
            col = []
            for _ in range(data["Height"]):
                col.append(None)
            cells.append(col)

        # Fill in the CellStates`
        for loc in data["Cells"].keys():
            x, y = eval(loc)
            cells[x][y] = cls.CellState.from_dict(data["Cells"][loc])

        return cls(width=data["Width"], height=data["Height"], cells=cells)


@dataclass
class PawnState:
    label: str
    is_ally: bool
    loc: Loc
    equipment: str
    combat_stats: "PawnState.CombatStats"
    health_stats: "PawnState.HealthStats"
    is_incapable: bool

    @dataclass
    class CombatStats:
        melee_DPS: float
        shooting_ACC: float
        move_speed: float

        def __iter__(self):
            yield ("melee_DPS", self.melee_DPS)
            yield ("shooting_ACC", self.shooting_ACC)
            yield ("move_speed", self.move_speed)

        @classmethod
        def from_dict(cls, data: Dict[str, float]) -> "PawnState.CombatStats":
            return cls(
                melee_DPS=data["MeleeDPS"],
                shooting_ACC=data["ShootingACC"],
                move_speed=data["MoveSpeed"],
            )

    @dataclass
    class HealthStats:
        pain_total: float
        blood_loss: float

        def __iter__(self):
            yield ("pain_total", self.pain_total)
            yield ("blood_loss", self.blood_loss)

        @classmethod
        def from_dict(cls, data: Dict[str, float]) -> "PawnState.HealthStats":
            return cls(pain_total=data["PainTotal"], blood_loss=data["BloodLoss"])

    def __iter__(self):
        yield ("label", self.label)
        yield ("is_ally", self.is_ally)
        yield ("loc", dict(self.loc))
        yield ("equipment", self.equipment)
        yield ("combat_stats", dict(self.combat_stats))
        yield ("health_stats", dict(self.health_stats))
        yield ("is_incapable", self.is_incapable)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PawnState":
        return cls(
            label=data["Label"],
            is_ally=data["IsAlly"],
            loc=Loc.from_dict(data["Loc"]),
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

    def __iter__(self):
        yield ("map_state", dict(self.map_state))
        yield ("pawn_states", {k: dict(v) for k, v in self.pawn_states.items()})
        yield ("tick", self.tick)
        yield ("game_ending", self.game_ending)

    @classmethod
    def from_dict(cls, data: Dict[str, dict | int | bool]) -> "GameState":
        map_state = MapState.from_dict(data["MapState"])
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
                    logger.info(
                        f"Collected game state at tick {cls.current_state.tick}\n"
                    )
                    logger.debug(
                        f"Map state (tick {cls.current_state.tick}): \n{to_json(cls.current_state.map_state, indent=2)}\n"
                    )
                    logger.debug(
                        f"Pawn state (tick {cls.current_state.tick}): \n{to_json(cls.current_state.pawn_states, indent=2)}\n"
                    )
                    logger.debug(
                        f"Game ending (tick {cls.current_state.tick}): {cls.current_state.game_ending}\n"
                    )
                    break
