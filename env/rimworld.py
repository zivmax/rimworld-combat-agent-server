import gymnasium as gym
import numpy as np
import logging
from threading import Thread
from numpy.typing import NDArray
from typing import Dict, List
from gymnasium.spaces import MultiDiscrete
from gymnasium import spaces

from utils.logger import get_cli_logger, get_file_logger
from utils.timestamp import timestamp
from utils.json import to_json
from .server import server, create_server_thread
from .state import StateCollector, CellState, MapState, PawnState, GameStatus, Loc
from .action import GameAction

logging_level = logging.INFO
f_logger = get_file_logger(
    __name__, f"env/logs/rimworld/{timestamp}.log", logging_level
)
cli_logger = get_cli_logger(__name__, logging_level)

logger = f_logger

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class RimWorldEnv(gym.Env):
    def __init__(self, options: Dict = None):

        self._pawns: Dict[str, PawnState] = None
        self._map: MapState = None
        self._allies: List[PawnState] = None
        self._enemies: List[PawnState] = None
        self._obstacles: Dict[str, List[Loc]] = {}
        self._last_obstacles: Dict[str, List[Loc]] = {}

        self._options: Dict = {
            "interval": options.get("interval", 1.0),
            "speed": options.get("speed", 1),
            "action_range": options.get("action_range", 4),
            "is_remote": options.get("is_remote", False),
            "rewarding": options.get(
                "rewarding",
                {
                    "original": 0,
                    "ally_down": -7,
                    "enemy_down": 10,
                    "ally_danger": 0.5,
                    "enemy_danger": -0.5,
                },
            ),
        }

        self._server_thread: Thread = create_server_thread(self._options["is_remote"])
        StateCollector.receive_state()

        self._update_all()

        self.action_space: Dict[int, MultiDiscrete] = {}
        for idx, ally in enumerate(self._allies, start=1):
            ally_space = MultiDiscrete(
                nvec=[
                    2 * self._options["action_range"] + 1,
                    2 * self._options["action_range"] + 1,
                ],
                start=np.array(
                    [-self._options["action_range"], -self._options["action_range"]],
                    dtype=np.int32,
                ),
            )
            self.action_space[idx] = ally_space
        self.action_space = spaces.Dict(self.action_space)
        self.observation_space = MultiDiscrete(
            [[8] * self._map.width] * self._map.height
        )

    def reset(self, seed=None, options: Dict = None):
        """Reset the environment to an initial state.
        This method resets the environment and returns the initial observation and info.
        The reset can be configured through the options parameter.
        Args:
            seed (int, optional): Random seed to use. Defaults to None.
            options (dict, optional): Configuration options for reset. Defaults to None.
                Supported options:
                    - interval (float): Time interval between ticks. Defaults to 1.0.
        Returns:
            tuple: A tuple containing:
                - observation: Initial environment observation
                - info: Additional information dictionary
        Note:
            This will send a reset signal to connected clients and reinitialize the state collector.
            The state will be updated and new observation will be generated based on the reset state.
        """
        if options is not None:
            self._options["interval"] = (
                options.get("interval", self._options["interval"])
                if options is not None
                else self._options["interval"]
            )
            self._options["speed"] = (
                options.get("speed", self._options["speed"])
                if options is not None
                else self._options["speed"]
            )

        # Validate options, interval be positive, speed between 0 - 4
        if self._options["interval"] <= 0:
            raise ValueError("Interval must be a positive number")
        if self._options["speed"] < 0 or self._options["speed"] > 4:
            raise ValueError("Speed must be between 0 and 4")

        message = {
            "Type": "Response",
            "Data": {
                "Action": None,
                "Reset": True,
                "Interval": self._options["interval"],
                "Speed": self._options["speed"],
            },
        }
        server.send_to_client(server.client, message)
        logger.info(f"Sent reset signal to clients at tick {StateCollector.state.tick}")

        super().reset(seed=seed)  # We need the following line to seed self.np_random
        StateCollector.reset()
        StateCollector.receive_state()

        logger.info(f"Env reset!")

        self._update_all()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: GameAction):
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = None

        # Transform the GameAction's relative target position refer to pawn itself, to absolute position on the map
        for ally in self._allies:
            if ally.label in action.pawn_actions.keys():
                action.pawn_actions[ally.label].x += ally.loc.x
                action.pawn_actions[ally.label].y += ally.loc.y

        message = {
            "Type": "Response",
            "Data": {
                "Action": dict(action),
                "Reset": False,
                "Interval": self._options["interval"],
                "Speed": self._options["speed"],
            },
        }

        server.send_to_client(server.client, message)

        StateCollector.receive_state()
        self._update_all()

        observation = self._get_obs()
        reward = self._get_reward(observation) if StateCollector.state.status == GameStatus.RUNNING else 0
        terminated = StateCollector.state.status != GameStatus.RUNNING
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        server.stop()
        super().close()

    def _update_allies(self):
        self._last_allies = self._allies.copy() if self._allies else None
        self._allies: list[PawnState] = [p for p in self._pawns.values() if p.is_ally]
        self._allies.sort(key=lambda x: x.label)

    def _update_enemies(self):
        self._last_enemies = self._enemies.copy() if self._enemies else None
        self._enemies: list[PawnState] = [
            p for p in self._pawns.values() if not p.is_ally
        ]
        self._enemies.sort(key=lambda x: x.label)

    def _update_all(self):
        self._pawns = StateCollector.state.pawns
        self._map = StateCollector.state.map
        self._update_allies()
        self._update_enemies()
        self._update_action_mask()

    def _Mannhatan_dist(self, loc1: Loc, loc2: Loc) -> float:
        return abs(loc1.x - loc2.x) + abs(loc1.y - loc2.y)

    def _Eclidean_dist(self, loc1: Loc, loc2: Loc) -> float:
        return np.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)

    def _clamp(self, loc: Loc) -> Loc:
        return Loc(
            max(min(loc.x, self._map.width - 1), 0),
            max(min(loc.y, self._map.height - 1), 0),
        )

    def _get_covers(self) -> List[CellState]:
        """Get all cover locations in the map.

        Returns a list of locations (Loc objects) where there are covers in the map.
        covers are defined as cells that contain either trees or walls.

        Returns:
            `List[Loc]`: A list of location objects representing positions of covers
        """
        covers = []
        for cell_row in self._map.cells:
            for cell in cell_row:
                if cell.is_tree or cell.is_wall:
                    covers.append(cell)
        return covers

    def _update_action_mask(self):
        """
        Returns a Dict action space where each key is an ally ID mapping to their movement space.
        Each ally's space is a MultiDiscrete for (x,y) movement across the entire map.

        The action mask is a tuple of invalid Loc positions per ally.
        """
        covers = self._get_covers()
        mask = {}

        # Collect invalid positions
        invalid_positions = []

        # Add covers
        for cover in covers:
            invalid_positions.append(cover.loc)

        # Add enemy positions
        for enemy in self._enemies:
            invalid_positions.append(enemy.loc)

        mask = tuple(invalid_positions)

        self.action_mask = mask
    def _search_neighbor_obstacles(self, loc: Loc, obs: NDArray, name:str) -> List[Loc]:
        neighbors = []
        for dx, dy in DIRECTIONS:
            neighbor = Loc(loc.x + dx, loc.y + dy)
            if 0 <= neighbor.x < self._map.width and 0 <= neighbor.y < self._map.height and obs[neighbor.x][neighbor.y] != 7:
                neighbors.append(neighbor)
        if name in self._obstacles.keys():
            self._last_obstacles[name] = self._obstacles[name]
        self._obstacles[name] = neighbors if neighbors else None
        return neighbors
    
    def _compare_obs(self, obs: List[Loc], last_obs: List[Loc]) -> NDArray:
        changes = []
        for current, last in zip(obs, last_obs):
            if current != last:
                changes.append(current)
        return changes
    def _get_obs(self) -> NDArray:
        """Gets the current observation of the game map as a 2D numpy array.

        The observation is represented as a grid where each cell contains an integer value:
        - 0: Empty cell
        - 1-3: Allied units (index + 1 corresponds to ally number)
        - 4-6: Enemy units (index + 4 corresponds to enemy number)
        - 7: Cover

        Returns:
            np.ndarray: A 2D numpy array of shape (height, width) containing integer values
                       representing the game state according to the above encoding scheme.
                       The array uses dtype=np.int32.

        Example:
            ```
            [[0, 1, 0],
             [7, 0, 4],
             [2, 0, 0]]
            ```
            Represents a 3x3 map with:
            - Ally 1 at (0,1)
            - Ally 2 at (2,0)
            - Enemy 1 at (1,2)
            - cover at (1,0)
        """
        grid = np.zeros((self._map.height, self._map.width), dtype=np.int8)
        covers = self._get_covers()

        for idx, ally in enumerate(self._allies, start=1):
            grid[ally.loc.x][ally.loc.y] = idx
        for idx, enemy in enumerate(self._enemies, start=4):
            grid[enemy.loc.x][enemy.loc.y] = idx
        for cover in covers:
            x, y = cover.loc.x, cover.loc.y
            grid[x][y] = 7
        return grid

    def _get_info(self):
        """
        Get current game information including the map state, pawn information and action space.

        Returns:
            tuple: A tuple containing three elements:
                - map (np.ndarray): Current game map state
                - pawn_in_ID (dict): Dictionary mapping pawn IDs to pawn objects, where:
                    - IDs 1-3 are reserved for allies
                    - IDs 4-6 are reserved for enemies
                - action_space (list): List of available actions
        """
        pawn_in_ID = {}
        for idx, ally in enumerate(self._allies, start=1):
            pawn_in_ID[idx] = ally
        for idx, enemy in enumerate(self._enemies, start=4):
            pawn_in_ID[idx] = enemy

        return {
            "map": self._map,
            "pawns": pawn_in_ID,
            "action_space": self.action_space,
            "action_mask": self.action_mask,
        }
    def _search_covers(self, loc: Loc) -> bool:
        covers = self._get_covers()
        for cover in covers:
            if cover.loc == loc:
                return True
        return False
    def _get_reward(self, obs:NDArray) -> float:
        """
        Calculate the reward based on the state of allies and enemies.

        The reward is determined by:
        1. For allies:
            - Deducts 10 points if an ally is incapacitated
            - Adds points based on ally's safety (inverse of danger)
        2. For enemies:
            - Adds 10 points if an enemy is incapacitated
            - Deducts points based on enemy's safety

        Returns:
             float: The calculated reward value. Positive values indicate favorable situations,
                     while negative values indicate unfavorable situations.
        """
        reward = self._options["rewarding"]["original"]
        for idx, ally in enumerate(self._allies):
            last_ally = self._last_allies[idx] if (self._last_allies and idx < len(self._last_allies)) else None

            if last_ally:
                if ally.is_incapable and not last_ally.is_incapable:
                    reward += self._options["rewarding"]["ally_down"]
                else:
                    reward += self._options["rewarding"]["ally_danger"] * (
                        ally.danger - last_ally.danger
                    )
                # search for close covers
                obstacles = self._search_neighbor_obstacles(ally.loc, obs, ally.label)
                difference = self._compare_obs(obstacles, self._last_obstacles[ally.label]) if ally.label in self._last_obstacles.keys() else obstacles
                if difference:
                    reward += self._options["rewarding"]["ally_cover"] * len(difference)
        for idx, enemy in enumerate(self._enemies):
            last_enemy = self._last_enemies[idx] if self._last_enemies else None
            if last_enemy:
                if enemy.is_incapable and not last_enemy.is_incapable:
                    reward += self._options["rewarding"]["enemy_down"]
                else:
                    reward += self._options["rewarding"]["enemy_danger"] * (
                        enemy.danger - last_enemy.danger
                    )
        return reward
