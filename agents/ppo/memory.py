from dataclasses import dataclass, astuple
from typing import List
import torch


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor

    def __iter__(self):
        return iter(astuple(self))


class PPOMemory:
    def __init__(self) -> None:
        self.transitions: List[Transition] = []

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        transition = Transition(
            state=state,
            action=action,
            log_prob=log_prob,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions = []
