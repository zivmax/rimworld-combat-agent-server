import numpy as np
from env.state import Loc


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Mannhatan_dist(loc1: Loc, loc2: Loc) -> float:
    return abs(loc1.x - loc2.x) + abs(loc1.y - loc2.y)


def Eclidean_dist(loc1: Loc, loc2: Loc) -> float:
    return np.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
