import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorWrapper
from collections import deque
from typing import Any, Dict, List, Tuple, Union


class VectorFrameStackObservation(VectorWrapper):
    def __init__(self, env: gym.vector.VectorEnv, stack_size: int = 4):
        """Stack n_stack frames for each environment in the vectorized env."""
        super().__init__(env)
        self.n_stack = stack_size

        # Create frame buffers for each env
        self.frames = [deque(maxlen=stack_size) for _ in range(self.num_envs)]

        # Modify observation space to add new stack dimension
        old_obs_space = env.single_observation_space
        new_shape = old_obs_space.shape[:-1] + (stack_size, old_obs_space.shape[-1])

        # Create new bounds matching the desired shape
        low = np.repeat(old_obs_space.low[np.newaxis, :], stack_size, axis=0)
        high = np.repeat(old_obs_space.high[np.newaxis, :], stack_size, axis=0)

        self.single_observation_space = gym.spaces.Box(
            low=low, high=high, dtype=old_obs_space.dtype, shape=new_shape
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environments and initialize frame stacks."""
        observations, info = self.env.reset(**kwargs)

        # Initialize frame stacks
        for idx in range(self.num_envs):
            for _ in range(self.n_stack):
                self.frames[idx].append(observations[idx])

        stacked_obs = self._get_stacked_observations()
        return stacked_obs, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Execute step and update frame stacks."""
        observations, rewards, terminateds, truncateds, infos = self.env.step(actions)

        # Update frame stacks
        for idx in range(self.num_envs):
            self.frames[idx].append(observations[idx])

            # Reset frame stack if episode ends
            if terminateds[idx] or truncateds[idx]:
                self.frames[idx].clear()
                for _ in range(self.n_stack):
                    self.frames[idx].append(observations[idx])

        stacked_obs = self._get_stacked_observations()
        return stacked_obs, rewards, terminateds, truncateds, infos

    def _get_stacked_observations(self) -> np.ndarray:
        """Convert frame stacks to stacked observations with new dimension."""
        stacked = []
        for idx in range(self.num_envs):
            # Stack frames along a new dimension
            env_frames = np.stack(list(self.frames[idx]), axis=-2)
            stacked.append(env_frames)
        return np.array(stacked)
