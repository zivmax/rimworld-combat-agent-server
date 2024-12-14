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
