import gymnasium as gym
import numpy as np
import logging
import signal
import sys
from functools import partial
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Optional
from gymnasium import spaces
from time import sleep
from dataclasses import dataclass, field

from utils.logger import get_cli_logger, get_file_logger
from utils.timestamp import timestamp
from utils.json import to_json
from .server import GameServer
from .state import StateCollector, CellState, MapState, PawnState, GameStatus, Loc
from .action import GameAction, PawnAction
from .game import Game, GameOptions
from .const import RESTART_INTERVAL, RIMWORLD_LOGGING_LEVEL

logging_level = RIMWORLD_LOGGING_LEVEL
f_logger = get_file_logger(
    __name__, f"env/logs/rimworld/{timestamp}.log", logging_level
)
cli_logger = get_cli_logger(__name__, logging_level)

logger = f_logger


def register_keyboard_interrupt(env: gym.Env):
    def handle_keyboard_interrupt(env: gym.Env, signum, frame):
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, partial(handle_keyboard_interrupt, env))


@dataclass
class EnvOptions:
    @dataclass
    class Rewarding:
        original: int = 0
        ally_defeated: int = 0
        enemy_defeated: int = 0
        ally_danger: float = 0
        enemy_danger: float = 0
        invalid_action: int = 0
        remain_still: int = 0
        win: int = 0
        lose: int = 0

    interval: float = 1.0
    speed: int = 1
    action_range: int = 4
    max_steps: Optional[int] = None
    is_remote: bool = False
    remain_still_threshold: int = 100
    rewarding: Rewarding = field(default_factory=Rewarding)
    game: GameOptions = field(default_factory=GameOptions)


class RimWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "headless"]}

    def __init__(
        self,
        options: EnvOptions = None,
        addr: str = "localhost",
        render_mode: str = "headless",
        port: int = 10086,
        bootsleep: int = 0,
    ):
        super().__init__()

        sleep(bootsleep)

        if render_mode not in ["headless", "human"]:
            raise ValueError(
                "Invalid render mode. Must be either 'headless' or 'human'."
            )

        self._reset_times: int = 0
        self._pawns: Dict[str, PawnState] = None
        self._map: MapState = None
        self._allies: Tuple[PawnState] = None
        self._allies_prev: Tuple[PawnState] = None
        self._enemies: Tuple[PawnState] = None
        self._enemies_prev: Tuple[PawnState] = None
        self._actions_prev: Dict[str, PawnAction] = None
        self._allies_still_count: Dict[str, int] = {}
        self._valid_positions: Tuple[Loc] = None
        self._valid_positions_prev: Tuple[Loc] = None
        self._steped_times: int = 0
        self._render_mode = render_mode
        self._game: Game = None

        self._options = options if options else EnvOptions()

        port = GameServer.find_available_port(port)
        self._server_thread, self._server = GameServer.create_server_thread(
            self._options.is_remote,
            port=port,
        )

        if self._render_mode == "headless":
            self._options.game.server_port = port
            self._options.game.server_addr = addr
            self._game = Game(
                game_path="/mnt/game/RimWorldLinux",
                options=self._options.game,
            )

            self._game.launch()

        StateCollector.receive_state(self._server)

        self._update_all()

        self.action_space: Dict[int, spaces.Box] = {}
        for idx, ally in enumerate(self._allies, start=1):
            ally_space = spaces.Box(
                low=-self._options.action_range,
                high=self._options.action_range,
                shape=(2,),
                dtype=np.int8,
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
        Returns:
            tuple: A tuple containing:
                - observation: Initial environment observation
                - info: Additional information dictionary
        Note:
            This will send a reset signal to connected clients and reinitialize the state collector.
            The state will be updated and new observation will be generated based on the reset state.
        """

        message = {
            "Type": "Response",
            "Data": {
                "Action": None,
                "Reset": True,
            },
        }
        self._server.send_to_client(message)
        logger.info(f"Sent reset signal to clients at tick {StateCollector.state.tick}")

        super().reset(
            seed=seed, options=options
        )  # We need the following line to seed self.np_random

        StateCollector.reset()
        while not StateCollector.receive_state(self._server):
            logger.warning(f"Timeout to reset the game, restarting the game.")
            self._game.restart()
            StateCollector.reset()
            sleep(1)
        else:
            logger.info(f"Reset the client game.")
            self._reset_times = 0

        logger.info(f"Env reset!")
        self._reset_times += 1
        self._steped_times = 0

        if self._reset_times >= RESTART_INTERVAL:
            self._game.restart()
            StateCollector.reset()
            while not StateCollector.receive_state(self._server):
                logger.warning(f"Timeout to restart the game, restarting the game.")
                self._game.restart()
                StateCollector.reset()
                sleep(1)
            else:
                logger.info(f"Restarted the client game.")
                self._reset_times = 0

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
                x=int(action[idx][0] + ally.loc.x),
                y=int(action[idx][1] + ally.loc.y),
            )

        game_action = GameAction(pawn_actions)

        message = {
            "Type": "Response",
            "Data": {
                "Action": dict(game_action),
                "Reset": False,
            },
        }

        self._server.send_to_client(message)
        logger.debug(
            f"Sent action to clients at tick {StateCollector.state.tick}: {to_json(game_action)}"
        )
        StateCollector.receive_state(self._server)

        self._actions_prev = pawn_actions
        self._update_all()

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = StateCollector.state.status != GameStatus.RUNNING
        truncated = (
            self._steped_times >= self._options.max_steps
            if self._options.max_steps is not None
            else False
        )
        info = self._get_info()
        self._steped_times += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        self._server.stop()
        if self._render_mode == "headless":
            self._game.shutdown()
        super().close()

    def _update_allies(self):
        self._allies_prev = self._allies
        self._allies: Tuple[PawnState] = tuple(
            sorted(
                [p for p in self._pawns.values() if p.is_ally], key=lambda x: x.label
            )
        )

        # Initialize still counts for new allies
        for ally in self._allies:
            self._allies_still_count.setdefault(ally.label, 0)

        # Update still counts based on previous positions
        if self._allies_prev:
            for ally_prev, ally_cur in zip(self._allies_prev, self._allies):
                if ally_prev.label == ally_cur.label:
                    if ally_prev.loc == ally_cur.loc:
                        self._allies_still_count[ally_cur.label] += 1
                    else:
                        self._allies_still_count[ally_cur.label] = 0

    def _update_enemies(self):
        self._enemies_prev = self._enemies
        self._enemies: Tuple[PawnState] = tuple(
            sorted(
                [p for p in self._pawns.values() if not p.is_ally],
                key=lambda x: x.label,
            )
        )

    def _update_all(self):
        self._pawns = StateCollector.state.pawns
        self._map = StateCollector.state.map
        self._update_allies()
        self._update_enemies()
        self._update_valid_position()

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

    def _update_valid_position(self):
        """
        Returns a Dict action space where each key is an ally ID mapping to their movement space.
        Each ally's space is a MultiDiscrete for (x,y) movement across the entire map.

        The action mask is a tuple of invalid Loc positions per ally.
        """
        self._valid_positions_prev = self._valid_positions
        covers = self._get_covers()

        # Collect invalid positions
        valid_positions = []
        for x in range(self._map.width):
            for y in range(self._map.height):
                valid_positions.append(Loc(x, y))

        # Add covers
        for cover in covers:
            if cover.loc in valid_positions:
                valid_positions.remove(cover.loc)

        # Add enemy positions
        for enemy in self._enemies:
            if enemy.loc in valid_positions:
                valid_positions.remove(enemy.loc)

        valid_positions = tuple(valid_positions)

        self._valid_positions = valid_positions

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
        ally_positions = np.zeros((self._map.height, self._map.width), dtype=np.uint8)
        enemy_positions = np.zeros((self._map.height, self._map.width), dtype=np.uint8)
        cover_positions = np.zeros((self._map.height, self._map.width), dtype=np.uint8)
        aiming_layer = np.zeros((self._map.height, self._map.width), dtype=np.uint8)
        status_layer = np.zeros((self._map.height, self._map.width), dtype=np.uint8)
        danger_layer = np.zeros((self._map.height, self._map.width), dtype=np.uint8)

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
            # Return layers as separate arrays in a tuple
            grid = np.array(
                [
                    ally_positions,
                    enemy_positions,
                    cover_positions,
                    aiming_layer,
                    status_layer,
                    danger_layer,
                ]
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
            "action_valid": self._valid_positions,
        }

    def _get_reward(self) -> float:
        """Calculate the total reward based on game state."""
        reward = self._options.rewarding.original

        reward += self._calculate_allies_reward()
        reward += self._calculate_enemies_reward()
        if StateCollector.state.status == GameStatus.WIN:
            reward += self._options.rewarding.win
        elif StateCollector.state.status == GameStatus.LOSE:
            reward += self._options.rewarding.lose

        return reward

    def _calculate_allies_reward(self) -> float:
        """Calculate reward component for allies."""
        reward = 0
        for idx, ally in enumerate(self._allies):
            reward += self._calculate_position_penalty(ally)
            reward += self._calculate_ally_state_change(idx, ally)
            reward += self._calculate_still_penalty(ally)
        return reward

    def _calculate_enemies_reward(self) -> float:
        """Calculate reward component for enemies."""
        reward = 0
        for idx, enemy in enumerate(self._enemies):
            reward += self._calculate_enemy_state_change(idx, enemy)
        return reward

    def _calculate_position_penalty(self, ally: PawnState) -> float:
        """Calculate penalty for invalid position."""
        prev_action_loc = Loc(
            self._actions_prev[ally.label].x, self._actions_prev[ally.label].y
        )
        if self._valid_positions_prev:
            if prev_action_loc not in self._valid_positions_prev:
                return self._options.rewarding.invalid_action
        return 0

    def _calculate_ally_state_change(self, idx: int, ally: PawnState) -> float:
        """Calculate reward based on ally's state changes."""
        if not self._allies_prev:
            return 0

        ally_prev = self._allies_prev[idx]
        if ally.is_incapable and not ally_prev.is_incapable:
            return self._options.rewarding.ally_defeated

        return self._options.rewarding.ally_danger * abs(ally.danger - ally_prev.danger)

    def _calculate_enemy_state_change(self, idx: int, enemy: PawnState) -> float:
        """Calculate reward based on enemy's state changes."""
        if not self._enemies_prev:
            return 0

        enemy_prev = self._enemies_prev[idx]
        if enemy.is_incapable and not enemy_prev.is_incapable:
            return self._options.rewarding.enemy_defeated

        return self._options.rewarding.enemy_danger * abs(
            enemy.danger - enemy_prev.danger
        )

    def _calculate_still_penalty(self, ally: PawnState) -> float:
        """Calculate penalty for remaining still too long."""

        excess_still_count = (
            self._allies_still_count[ally.label] - self._options.remain_still_threshold
        )

        if excess_still_count > 0:
            return self._options.rewarding.remain_still
        return 0
