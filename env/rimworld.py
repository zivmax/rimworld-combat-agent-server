import gymnasium as gym
import numpy as np
import logging
from threading import Thread
from numpy.typing import NDArray
from typing import Dict, List
from gymnasium import spaces

from utils.logger import get_cli_logger, get_file_logger
from utils.timestamp import timestamp
from utils.json import to_json
from .server import server, create_server_thread
from .state import StateCollector, CellState, MapState, PawnState, GameStatus, Loc
from .action import GameAction, PawnAction

logging_level = logging.INFO
f_logger = get_file_logger(
    __name__, f"env/logs/rimworld/{timestamp}.log", logging_level
)
cli_logger = get_cli_logger(__name__, logging_level)

logger = f_logger


class RimWorldEnv(gym.Env):
    def __init__(self, options: Dict = None):

        self._pawns: Dict[str, PawnState] = None
        self._map: MapState = None
        self._allies: List[PawnState] = None
        self._allies_prev: List[PawnState] = None
        self._enemies: List[PawnState] = None
        self._enemies_prev: List[PawnState] = None
        self._covers: Dict[str, List[Loc]] = {}
        self._covers_prev: Dict[str, List[Loc]] = {}

        self._options: Dict = {
            "interval": options.get("interval", 1.0),
            "speed": options.get("speed", 1),
            "action_range": options.get("action_range", 4),
            "is_remote": options.get("is_remote", False),
            "rewarding": options.get(
                "rewarding",
                {
                    "original": 0,
                    "ally_defeated": -7,
                    "enemy_defeated": 10,
                    "ally_danger": 0.5,
                    "enemy_danger": -0.5,
                },
            ),
        }

        self._server_thread: Thread = create_server_thread(self._options["is_remote"])
        StateCollector.receive_state()

        self._update_all()

        self.action_space: Dict[int, spaces.MultiDiscrete] = {}
        for idx, ally in enumerate(self._allies, start=1):
            ally_space = spaces.MultiDiscrete(
                nvec=[
                    2 * self._options["action_range"] + 1,
                    2 * self._options["action_range"] + 1,
                ],
                start=np.array(
                    [-self._options["action_range"], -self._options["action_range"]],
                    dtype=np.int16,
                ),
            )
            self.action_space[idx] = ally_space
        self.action_space = spaces.Dict(self.action_space)

        """
        Observation space has 6 layers:
        1. Ally positions layer (0 to len(allies), uint8)
        2. Enemy positions layer (0 to len(enemies), uint8)
        3. Cover positions layer (0-1, uint8)
        4. Aiming layer (0-1, uint8)
        5. Status layer (0-1, uint8)
        6. Danger layer (0-100, uint8)
        """
        self.observation_space = spaces.Box(
            low=0,
            high=np.array(
                [
                    [[len(self._allies)] * self._map.width] * self._map.height,
                    [[len(self._enemies)] * self._map.width] * self._map.height,
                    [[1] * self._map.width] * self._map.height,
                    [[1] * self._map.width] * self._map.height,
                    [[1] * self._map.width] * self._map.height,
                    [[100] * self._map.width] * self._map.height,
                ]
            ),
            shape=(6, self._map.height, self._map.width),
            dtype=np.uint8,
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

    def step(self, action: Dict):
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = None

        pawn_actions = {}
        for idx, ally in enumerate(self._allies, start=1):
            pawn_actions[ally.label] = PawnAction(
                label=ally.label,
                x=action[idx][0] + ally.loc.x,
                y=action[idx][1] + ally.loc.y,
            )

        game_action = GameAction(pawn_actions)

        message = {
            "Type": "Response",
            "Data": {
                "Action": dict(game_action),
                "Reset": False,
                "Interval": self._options["interval"],
                "Speed": self._options["speed"],
            },
        }

        server.send_to_client(server.client, message)

        StateCollector.receive_state()
        self._update_all()

        observation = self._get_obs()
        reward = (
            self._get_reward(observation)
            if StateCollector.state.status == GameStatus.RUNNING
            else 0
        )
        terminated = StateCollector.state.status != GameStatus.RUNNING
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        server.stop()
        super().close()

    def _update_allies(self):
        self._allies_prev = self._allies.copy() if self._allies else None
        self._allies: list[PawnState] = [p for p in self._pawns.values() if p.is_ally]
        self._allies.sort(key=lambda x: x.label)

    def _update_enemies(self):
        self._enemies_prev = self._enemies.copy() if self._enemies else None
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

    def _compare_obs(self, obs: List[Loc], last_obs: List[Loc]) -> NDArray:
        if not last_obs:
            return obs
        changes = []
        for current, last in zip(obs, last_obs):
            if current != last:
                changes.append(current)
        return changes

    def _get_obs(self) -> NDArray:
        """Gets the current observation of the game map as a stacked array of 2D grids.

        The observation consists of 6 layers stacked along axis 0:
        1. Ally positions layer:
            - 0: Empty cell
            - 1 to len(allies): Allied units
        2. Enemy positions layer:
            - 0: Empty cell
            - 1 to len(enemies): Enemy units
        3. Cover positions layer:
            - 0: No cover
            - 1: Cover present
        4. Aiming layer:
            - 0: Not aiming
            - 1: Aiming
        5. Status layer:
            - 0: Incapable
            - 1: Capable
        6. Danger layer: Integer values 0-100 representing danger level
           (converted from float 0-1)

        Returns:
            NDArray: A 6D numpy array of shape (6, height, width) containing the stacked observation layers.
                    All layers use dtype=np.int8.
        """
        # Create separate layers
        ally_positions = np.zeros((self._map.height, self._map.width), dtype=np.int8)
        enemy_positions = np.zeros((self._map.height, self._map.width), dtype=np.int8)
        cover_positions = np.zeros((self._map.height, self._map.width), dtype=np.int8)
        aiming_layer = np.zeros((self._map.height, self._map.width), dtype=np.int8)
        status_layer = np.zeros((self._map.height, self._map.width), dtype=np.int8)
        danger_layer = np.zeros((self._map.height, self._map.width), dtype=np.int8)

        # Fill ally positions
        for idx, ally in enumerate(self._allies, start=1):
            x, y = ally.loc.x, ally.loc.y
            ally_positions[x][y] = idx
            aiming_layer[x][y] = 1 if ally.is_aiming else 0
            status_layer[x][y] = 0 if ally.is_incapable else 1
            danger_layer[x][y] = int(ally.danger * 100)

        # Fill enemy positions
        for idx, enemy in enumerate(self._enemies, start=1):
            x, y = enemy.loc.x, enemy.loc.y
            enemy_positions[x][y] = idx
            aiming_layer[x][y] = 1 if enemy.is_aiming else 0
            status_layer[x][y] = 0 if enemy.is_incapable else 1
            danger_layer[x][y] = int(enemy.danger * 100)

        # Fill cover positions
        for cover in self._get_covers():
            x, y = cover.loc.x, cover.loc.y
            cover_positions[x][y] = 1

        # Stack layers into a single array
        grid = np.stack(
            [
                ally_positions,
                enemy_positions,
                cover_positions,
                aiming_layer,
                status_layer,
                danger_layer,
            ],
            axis=0,
        )
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

    def _get_reward(self, obs: NDArray) -> float:
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
        match StateCollector.state.status:
            case GameStatus.RUNNING:
                ally_step_defeated_num = 0
                enemy_step_defeated_num = 0
                for idx, ally in enumerate(self._allies):
                    ally_prev = self._allies_prev[idx] if self._allies_prev else None

                    if ally_prev:
                        if ally.is_incapable and not ally_prev.is_incapable:
                            reward += self._options["rewarding"]["ally_defeated"]
                        else:
                            reward += self._options["rewarding"]["ally_danger"] * abs(
                                ally.danger - ally_prev.danger
                            )
                    if ally.is_incapable:
                        ally_step_defeated_num += 1
                for idx, enemy in enumerate(self._enemies):
                    enemy_prev = self._enemies_prev[idx] if self._enemies_prev else None

                    if enemy_prev:
                        if enemy.is_incapable and not enemy_prev.is_incapable:
                            reward += self._options["rewarding"]["enemy_defeated"]
                        else:
                            reward += self._options["rewarding"]["enemy_danger"] * abs(
                                enemy.danger - enemy_prev.danger
                            )
                    if enemy.is_incapable:
                        enemy_step_defeated_num += 1

            case GameStatus.WIN:
                reward += self._options["rewarding"]["win"]

            case GameStatus.LOSE:
                reward += self._options["rewarding"]["lose"]

        return reward
