from collections import deque
from typing import Any, Final, List, Optional, Tuple, Union
from copy import deepcopy

import numpy as np
from gymnasium.vector import VectorEnv, VectorWrapper
from gymnasium.core import ObsType, WrapperObsType, WrapperActType
from gymnasium.vector.utils import (
    batch_space,
    concatenate,
    create_empty_array,
)
from gymnasium.wrappers.utils import create_zero_array


class FrameStackObservation(VectorWrapper):
    """Stacks the observations from the last ``N`` time steps in a rolling manner for vectorized environments.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Users have options for the padded observation used:

     * "reset" (default) - The reset value is repeated
     * "zero" - A "zero"-like instance of the observation space
     * custom - An instance of the observation space

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStackObservation
        >>> env = gym.make_vec("CarRacing-v3", num_envs=4)
        >>> env = FrameStackObservation(env, stack_size=4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 4, 96, 96, 3)

    Example with different padding observations:
        >>> env = gym.make_vec("CartPole-v1", num_envs=4)
        >>> env.reset(seed=123)
        (array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]], dtype=float32), {})
        >>> stacked_env = FrameStackObservation(env, 3)   # the default is padding_type="reset"
        >>> stacked_env.reset(seed=123)
        (array([[[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
               [[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
               [[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
               [[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]]], dtype=float32), {})

        >>> stacked_env = FrameStackObservation(env, 3, padding_type="zero")
        >>> stacked_env.reset(seed=123)
        (array([[[ 0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
               [[ 0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
               [[ 0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
               [[ 0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ,  0.        ],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]]], dtype=float32), {})
        >>> stacked_env = FrameStackObservation(env, 3, padding_type=np.array([1, -1, 0, 2], dtype=np.float32))
        >>> stacked_env.reset(seed=123)
        (array([[[ 1.        , -1.        ,  0.        ,  2.        ],
                [ 1.        , -1.        ,  0.        ,  2.        ],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
               [[ 1.        , -1.        ,  0.        ,  2.        ],
                [ 1.        , -1.        ,  0.        ,  2.        ],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
               [[ 1.        , -1.        ,  0.        ,  2.        ],
                [ 1.        , -1.        ,  0.        ,  2.        ],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
               [[ 1.        , -1.        ,  0.        ,  2.        ],
                [ 1.        , -1.        ,  0.        ,  2.        ],
                [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]]], dtype=float32), {})
    """

    def __init__(
        self,
        env: VectorEnv,
        stack_size: int,
        *,
        padding_type: Union[str, ObsType] = "reset",
    ):
        """Observation wrapper that stacks the observations in a rolling manner for vectorized environments.

        Args:
            env: The vectorized environment to apply the wrapper
            stack_size: The number of frames to stack.
            padding_type: The padding type to use when stacking the observations, options: "reset", "zero", custom obs
        """
        super().__init__(env)

        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(
                f"The stack_size is expected to be an integer, actual type: {type(stack_size)}"
            )
        if not 0 < stack_size:
            raise ValueError(
                f"The stack_size needs to be greater than zero, actual value: {stack_size}"
            )
        if isinstance(padding_type, str) and (
            padding_type == "reset" or padding_type == "zero"
        ):
            self.padding_values: ObsType = [
                create_zero_array(env.observation_space) for _ in range(self.num_envs)
            ]
        elif padding_type in env.single_observation_space:
            self.padding_values = [padding_type for _ in range(self.num_envs)]
            padding_type = "_custom"
        else:
            if isinstance(padding_type, str):
                raise ValueError(  # we are guessing that the user just entered the "reset" or "zero" wrong
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r}"
                )
            else:
                raise ValueError(
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r} not an instance of env observation ({env.single_observation_space})"
                )

        self.single_observation_space = batch_space(
            env.single_observation_space, n=stack_size
        )
        self.stack_size: Final[int] = stack_size
        self.padding_type: Final[str] = padding_type

        self.obs_queues = [
            deque(
                [self.padding_values for _ in range(self.stack_size)],
                maxlen=self.stack_size,
            )
            for _ in range(self.num_envs)
        ]
        self.stacked_obses = [
            create_empty_array(env.single_observation_space, n=self.stack_size)
            for _ in range(self.num_envs)
        ]

    def step(
        self, actions: WrapperActType
    ) -> Tuple[WrapperObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            actions: The actions to step through the environment with

        Returns:
            Stacked observations, rewards, terminateds, truncateds, and infos from the environment
        """
        obses, rewards, terminateds, truncateds, infos = self.env.step(actions)
        for i in range(self.num_envs):
            self.obs_queues[i].append(obses[i])

        updated_obses = [
            deepcopy(
                concatenate(
                    self.env.single_observation_space,
                    self.obs_queues[i],
                    self.stacked_obses[i],
                )
            )
            for i in range(self.num_envs)
        ]
        return updated_obses, rewards, terminateds, truncateds, infos

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[WrapperObsType, dict[str, Any]]:
        """Reset the environment, returning the stacked observation and info.

        Args:
            seed: The environment seed
            options: The reset options

        Returns:
            The stacked observations and info
        """
        obses, info = self.env.reset(seed=seed, options=options)

        if self.padding_type == "reset":
            self.padding_values = obses
        for i in range(self.num_envs):
            for _ in range(self.stack_size - 1):
                self.obs_queues[i].append(self.padding_values[i])
            self.obs_queues[i].append(obses[i])

        updated_obses = [
            deepcopy(
                concatenate(
                    self.env.single_observation_space,
                    self.obs_queues[i],
                    self.stacked_obses[i],
                )
            )
            for i in range(self.num_envs)
        ]

        return updated_obses, info
