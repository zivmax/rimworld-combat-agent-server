import gym
from gym import spaces
import numpy as np
import random
from typing import List
from state_utils import Sutil
from utils.logger import logger

ALLY_NUM = 3
ENEMY_NUM = 3
HEIGHT = 30
WIDTH = 30
class RimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=10):
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
        return state.flatten(), pawn_map

    def reset(self):
        util = Sutil()
        self.allies, self.enemies = util.get_allies_enemies()
        self.last_step_allies, self.last_step_enemies = self.allies.copy(), self.enemies.copy()
        self.obstacles = util.get_obstacles()
        self.steps = 0
        self.done = False
        self.reward = 0

        return self._get_state()
    def _eucild_dist(self, loc1:tuple, loc2:tuple)->float:
        return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5
    def _clamp(self, loc:tuple)->tuple:
        return (max(min(loc[0], HEIGHT - 1), 0), 
                max(min(loc[1], WIDTH - 1), 0))

    def step(self, ally_action:List[tuple]):
        if self.done:
            logger.info("Game has ended, please reset the env")
            return

        self.steps += 1
        self.reward = 0
        down_ally = 0
        for idx in range(ALLY_NUM):
            ally = self.allies[idx]
            last_step_ally = self.last_step_allies[idx]
            '''
            incap is equal to dead
            encourage avoid incap
            '''
            if ally.is_incapable:
                if not last_step_ally.is_incapable:
                    self.reward -= 10
                last_step_ally = ally.copy()
                down_ally += 1
                # lost
                if down_ally == ALLY_NUM:
                    self.done = True
                    self.reward -= 50
                    return self.reward, self.done

                continue

            '''
            encourage to keep an eye on incap rate
            '''





            target_pos = self._clamp(ally_action[idx])
            '''
            encourage only valid move
            '''
            if target_pos in self.obstacles:
                self.reward -= 5
            else:
                last_step_ally = ally.copy()
                ally.loc = target_pos
            '''
            encourage move more than 1 step
            '''
            if self._eucild_dist(ally.loc, last_step_ally.loc) > 1.2:
                self.reward += 2
            '''
            reserve:encourage the pawn with high speed to move frequently
            '''
            '''
            reserve:encourage the pawn with exceptionally high melee
            to do close attack
            '''

        # assume enemy does not move in one step
        down_enemy = 0
        for idx in range(ENEMY_NUM):
            enemy = self.enemies[idx]
            last_step_enemy = self.last_step_enemies[idx]
            '''
            incap is equal to dead
            encourage cause incap
            '''
            if enemy.is_incapable:
                if not last_step_enemy.is_incapable:
                    self.reward += 10
                last_step_enemy = enemy.copy()
                down_enemy += 1
                # win
                if down_enemy == ENEMY_NUM:
                    self.done = True
                    self.reward += 50
                    return self.reward, self.done
                continue

            '''
            encourage to keep an eye on incap rate
            '''




            last_step_enemy = enemy.copy()


        if self.steps >= self.max_steps:
            self.done = True

        return self._get_state(), self.reward, self.done

            