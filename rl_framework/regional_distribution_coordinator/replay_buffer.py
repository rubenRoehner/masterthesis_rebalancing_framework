from collections import deque
from typing import NamedTuple, List, Deque
import random
import torch


class Experience(NamedTuple):
    state: torch.Tensor
    action_indices: List[int]
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.capacity: int = capacity
        self.buffer: Deque[Experience] = deque([], maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action_indices: List[int],
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.append(Experience(state, action_indices, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
