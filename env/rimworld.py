import gymnasium as gym
import numpy as np
from threading import Thread
from numpy.typing import NDArray
from typing import Dict, List, Tuple
from gymnasium.spaces import MultiDiscrete
from gymnasium import spaces

from utils.logger import logger
from .server import server, create_server_thread
from .state import StateCollector, MapState, PawnState, GameStatus, Loc
from .action import GameAction
from metadata import REWARD, ACTION_SPACE_RADIUS_FACTOR

class RimWorldEnv(gym.Env):
    def __init__(self):
        self._server_thread: Thread = create_server_thread()

        self._pawns: Dict[str, PawnState] = None
        self._map: MapState = None
        self._allies: list[PawnState] = None
        self._enemies: list[PawnState] = None
        self.action_space = None
        self.action_mask = None

        StateCollector.receive_state()
        self._update_all()
        self.observation_space = MultiDiscrete(
            [[8] * self._map.width] * self._map.height
        )

    def reset(self, seed=None, options=None):
        message = {
            "Type": "Response",
            "Data": {"Action": None, "Reset": True},
        }
        server.send_to_client(server.client, message)
        logger.info(
            f"\tSent reset signal to clients at tick {StateCollector.state.tick}\n"
        )

        super().reset(seed=seed)  # We need the following line to seed self.np_random
        StateCollector.reset()
        StateCollector.receive_state()
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


        message = {
            "Type": "Response",
            "Data": {"Action": dict(action), "Reset": False},
        }

        server.send_to_client(server.client, message)

        StateCollector.receive_state()
        self._update_all()

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = StateCollector.state.status != GameStatus.RUNNING
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        server.stop()
        self._server_thread.join()
        super().close()

    def _update_allies(self):
        self._allies: list[PawnState] = [p for p in self._pawns.values() if p.is_ally]
        self._allies.sort(key=lambda x: x.label)

    def _update_enemies(self):
        self._enemies: list[PawnState] = [
            p for p in self._pawns.values() if not p.is_ally
        ]
        self._enemies.sort(key=lambda x: x.label)

    def _update_all(self):
        self._pawns = StateCollector.state.pawns
        self._map = StateCollector.state.map
        self._update_allies()
        self._update_enemies()
        self._update_action_space()


    def _clamp(self, loc: Loc) -> Loc:
        return Loc(
            max(min(loc.x, self._map.width - 1), 0),
            max(min(loc.y, self._map.height - 1), 0),
        )

    def _get_covers(self) -> List[Loc]:
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
                    covers.append(cell.loc)
        return covers

    def _update_action_space(self):
        """
        Returns a Dict action space where each key is an ally ID mapping to their movement space.
        Each ally's space is a MultiDiscrete for (x,y) movement within their max_move range.

        The action mask is now a tuple of invalid Loc positions per ally.
        """
        covers = self._get_covers()
        action_spaces = {}
        action_masks = {}

        for idx, ally in enumerate(self._allies, start=1):
            # Calculate max movement range
            max_move = int(ACTION_SPACE_RADIUS_FACTOR * ally.combat.move_speed * ally.health.moving)

            # Calculate ranges for this ally
            min_x = max(0, ally.loc.x - max_move)
            max_x = min(self._map.width, ally.loc.x + max_move + 1)
            min_y = max(0, ally.loc.y - max_move)
            max_y = min(self._map.height, ally.loc.y + max_move + 1)

            # Collect invalid positions
            invalid_positions = []

            # Add covers within range
            for obs in covers:
                if min_x <= obs.x < max_x and min_y <= obs.y < max_y:
                    invalid_positions.append(obs)

            # Add enemy positions within range
            for enemy in self._enemies:
                if min_x <= enemy.loc.x < max_x and min_y <= enemy.loc.y < max_y:
                    invalid_positions.append(enemy.loc)

            ally_space = MultiDiscrete(
                nvec=[max_x - min_x, max_y - min_y], start=np.array([min_x, min_y])
            )

            action_spaces[idx] = ally_space
            action_masks[idx] = tuple(invalid_positions)

        self.action_space = spaces.Dict(action_spaces)
        self.action_mask = action_masks

    def _get_obs(self) -> NDArray:
        """Gets the current observation of the game map as a 2D numpy array.

        The observation is represented as a grid where each cell contains an integer value:
        - 0: Empty cell
        - 1-3: Allied units (index + 1 corresponds to ally number)
        - 4-6: Enemy units (index + 4 corresponds to enemy number)
        - 7: cover/Wall

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
            x, y = cover.x, cover.y
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

    def _get_reward(self) -> float:
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
        reward = REWARD['original']
        for ally in self._allies:
            if ally.is_incapable:
                reward += REWARD['ally_down']
            else:
                reward += REWARD['ally_danger_ratio'] * (1 - ally.danger)
        for enemy in self._enemies:
            if enemy.is_incapable:
                reward += REWARD['enemy_down']
            else:
                reward += REWARD['enemy_danger_ratio'] * (1 - enemy.danger)
        return reward
