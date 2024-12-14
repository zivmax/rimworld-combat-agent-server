import gym
from gym import spaces
import numpy as np
import random
from state_utils import Sutil

ALLY_NUM = 3
ENEMY_NUM = 3
HEIGHT = 30
WIDTH = 30
class RimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=30):
        super(RimEnv, self).__init__()
        
        self.max_steps = max_steps

        self.action_space = spaces.MultiDiscrete([
            HEIGHT,
            WIDTH
        ])
        '''
        7: Empty
        0-2: Allies
        3-5: Enemies
        6: Obstacles
        '''
        self.observation_space = spaces.MultiDiscrete([8] * HEIGHT * WIDTH)
        self.reset()

    def _get_state(self):
        state = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
        pawn_map = {}
        for pos in self.obstacles:
            x, y = pos
            state[x][y] = 6
        for idx, ally in enumerate(self.allies):
            state[ally.loc.x][ally.loc.y] = idx
            pawn_map[idx] = ally
        for idx, enemy in enumerate(self.enemies):
            state[enemy.loc.x][enemy.loc.y] = idx + 3
            pawn_map[idx + 3] = enemy
        return state

    def reset(self):
        self._util = Sutil()
        self.allies, self.enemies = self._util.get_allies_enemies()
        self.obstacles = self._util.get_obstacles()
        self.steps = 0
        self.done = False
        self.reward = 0

        return self._get_state()
