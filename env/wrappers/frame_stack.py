from collections import deque
import gymnasium as gym
import numpy as np


class FrameStackEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int):
        super(FrameStackEnv, self).__init__(env)
        self.frames = deque([], maxlen=k)

        # Update the observation space to have k times the original number of layers
        original_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.repeat(env.observation_space.high[np.newaxis, :], k, axis=0),
            shape=(original_shape[0] * k, original_shape[1], original_shape[2]),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.frames.maxlen):
            self.frames.append(ob)
        return np.stack(self.frames, axis=0)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return np.stack(self.frames, axis=0), reward, done, info
