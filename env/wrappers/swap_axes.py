from typing import List

import numpy as np
from gymnasium import Env, ObservationWrapper
from gymnasium.core import ObsType
from gymnasium import spaces


class SwapObservationAxes(ObservationWrapper):
    def __init__(
        self,
        env: Env,
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
        old_shape = self.observation_space.shape
        new_shape = list(old_shape)
        new_shape[self.axis1], new_shape[self.axis2] = (
            new_shape[self.axis2],
            new_shape[self.axis1],
        )

        if isinstance(self.observation_space, spaces.Box):
            self.observation_space = spaces.Box(
                low=self.observation_space.low.swapaxes(self.axis1, self.axis2),
                high=self.observation_space.high.swapaxes(self.axis1, self.axis2),
                dtype=self.observation_space.dtype,
                shape=new_shape,
            )
        else:
            raise ValueError(
                "SwapAxes only supports Box observation spaces at the moment"
            )

    def observation(self, observations: ObsType) -> ObsType:
        """Swap the specified axes of the observation.

        Args:
            observations: The observation from the environment

        Returns:
            The observation with swapped axes
        """
        # Adding 1 to the axes to account for the batch dimension
        return np.swapaxes(observations, self.axis1, self.axis2)
