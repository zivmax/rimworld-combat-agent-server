from collections import deque
from typing import Deque, List, Tuple, Any

import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity: int = capacity
        self.buffer: Deque[Any] = deque(maxlen=capacity)
        self.priorities: Deque[float] = deque(maxlen=capacity)
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, experience: Any, priority: float) -> None:
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(
        self, batch_size: int, beta: float
    ) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        assert len(self.buffer) >= batch_size

        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]

        # Importance-sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: List[float]) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
