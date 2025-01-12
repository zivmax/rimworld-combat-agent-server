from typing import List

import numpy as np
from gymnasium.vector import VectorEnv, VectorObservationWrapper
from gymnasium.core import ObsType
from gymnasium import spaces
from gymnasium.vector.utils import batch_space


class SwapObservationAxes(VectorObservationWrapper):
    def __init__(
        self,
        env: VectorEnv,
        swap: List[int],
    ):
        """Initialize the SwapAxes wrapper.

        Args:
            env: The vector environment to wrap
            swap: List of two integers specifying the axes to swap
        """
        super().__init__(env)

        if len(swap) != 2:
            raise ValueError("swap_axes must contain exactly 2 integers")

        self.axis1, self.axis2 = swap[0], swap[1]

        # Update the observation space
        old_shape = self.single_observation_space.shape
        new_shape = list(old_shape)
        new_shape[self.axis1], new_shape[self.axis2] = (
            new_shape[self.axis2],
            new_shape[self.axis1],
        )

        if isinstance(self.single_observation_space, spaces.Box):
            self.single_observation_space = spaces.Box(
                low=self.single_observation_space.low.swapaxes(self.axis1, self.axis2),
                high=self.single_observation_space.high.swapaxes(
                    self.axis1, self.axis2
                ),
                dtype=self.single_observation_space.dtype,
                shape=new_shape,
            )
        else:
            raise ValueError(
                "SwapAxes only supports Box observation spaces at the moment"
            )

        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

    def observations(self, observations: ObsType) -> ObsType:
        """Swap the specified axes of the observation.

        Args:
            observations: The observation from the environment

        Returns:
            The observation with swapped axes
        """
        # Adding 1 to the axes to account for the batch dimension
        return np.swapaxes(observations, self.axis1 + 1, self.axis2 + 1)
