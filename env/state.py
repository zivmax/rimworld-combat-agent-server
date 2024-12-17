from dataclasses import dataclass
from typing import Dict, List

from env.server import server
from utils.logger import logger
from utils.json import to_json
from utils.math import sigmoid


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


class GameStatus:
    RUNNING = "RUNNING"
    LOSE = "LOSE"
    WIN = "WIN"


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


@dataclass
class MapState:
    width: int
    height: int
    cells: List[List["CellState"]]

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
            cells[x][y] = CellState.from_dict(data["Cells"][loc])

        return cls(width=data["Width"], height=data["Height"], cells=cells)


@dataclass
class PawnState:
    label: str
    loc: Loc
    is_ally: bool
    is_incapable: bool
    is_aiming: bool
    equipment: str
    combat: "CombatStats"
    health: "HealthStats"

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
        pain: float
        bleed: float
        moving: float
        conciousness: float

        def __iter__(self):
            yield ("pain", self.pain)
            yield ("bleed", self.bleed)
            yield ("moving", self.moving)
            yield ("conciousness", self.conciousness)

        @classmethod
        def from_dict(cls, data: Dict[str, float]) -> "PawnState.HealthStats":
            return cls(
                pain=data["PainTotal"],
                bleed=data["BloodLoss"],
                moving=data["MoveAbility"],
                conciousness=data["Consciousness"],
            )

    @property
    def danger(self) -> float:
        return sigmoid(
            1 - self.health.conciousness + self.health.pain + self.health.bleed
        )

    def __iter__(self):
        yield ("label", self.label)
        yield ("loc", dict(self.loc))
        yield ("is_ally", self.is_ally)
        yield ("is_incapable", self.is_incapable)
        yield ("is_aiming", self.is_aiming)
        yield ("equipment", self.equipment)
        yield ("combat_stats", dict(self.combat))
        yield ("health_stats", dict(self.health))

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PawnState":
        return cls(
            label=data["Label"],
            loc=Loc.from_dict(data["Loc"]),
            is_ally=data["IsAlly"],
            is_incapable=data["IsIncapable"],
            is_aiming=data["IsAiming"],
            equipment=data["Equipment"],
            combat=cls.CombatStats.from_dict(data["CombatStats"]),
            health=cls.HealthStats.from_dict(data["HealthStats"]),
        )


@dataclass
class GameState:
    map: MapState
    pawns: Dict[str, PawnState]
    status: "GameStatus"
    tick: int

    def __iter__(self):
        yield ("map", dict(self.map))
        yield ("pawns", {k: dict(v) for k, v in self.pawns.items()})
        yield ("tick", self.tick)
        yield ("status", self.status)

    @classmethod
    def from_dict(cls, data: Dict[str, dict | int | str]) -> "GameState":
        map_state = MapState.from_dict(data["MapState"])
        pawn_states = {}

        for pawn_label, pawn_data in data["PawnStates"].items():
            pawn_states[pawn_label] = PawnState.from_dict(pawn_data)

        return cls(
            map=map_state,
            pawns=pawn_states,
            status=data["Status"],
            tick=data["Tick"],
        )


class StateCollector:
    state = None

    @classmethod
    def reset(cls) -> None:
        cls.state = None

    @classmethod
    def is_new_state(cls, tick: int) -> bool:
        return cls.state is None or tick > cls.state.tick

    @classmethod
    def receive_state(cls) -> None:
        while True:
            if server.client is None:
                cls.reset()
                continue
            if server.message_queue.qsize() > 0:
                # Peek at message without removing it
                message = server.message_queue.queue[0]
                # Only get the message if it's a state message
                if "GameState" != message["Type"]:
                    continue

                message = server.message_queue.get()

                if not cls.is_new_state(message["Data"]["Tick"]):
                    continue

                cls.state = GameState.from_dict(message["Data"])

                logger.debug(f"Collected game state at tick {cls.state.tick}\n")
                logger.debug(
                    f"Map state (tick {cls.state.tick}): \n{to_json(cls.state.map, indent=2)}\n"
                )
                logger.debug(
                    f"Pawn state (tick {cls.state.tick}): \n{to_json(cls.state.pawns, indent=2)}\n"
                )
                logger.debug(
                    f"Game status (tick {cls.state.tick}): {cls.state.status}\n"
                )
                break
