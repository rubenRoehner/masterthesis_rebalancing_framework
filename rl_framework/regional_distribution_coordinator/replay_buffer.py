"""
replay_buffer.py

Replay buffer for storing experiences with prioritized sampling.
This buffer supports prioritized experience replay, which allows the agent to sample experiences based on their importance.
"""

from collections import deque
from typing import NamedTuple, List, Deque, Tuple
import numpy as np
import torch


class Experience(NamedTuple):
    """A single experience in the replay buffer.

    Attributes:
        state: current state as a torch.Tensor
        action_indices: list of action indices taken in the current state
        reward: reward received for the action taken
        next_state: next state as a torch.Tensor
        done: boolean indicating if the episode has ended
    """

    state: torch.Tensor
    action_indices: List[int]
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6) -> None:
        """Initialize the replay buffer.
        Args:
            capacity: max number of experiences to store
            alpha: how much prioritization is used (0 = uniform, 1 = full prioritization)
            eps: small constant so we never have zero priority
        """
        self.capacity = capacity
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.eps = eps
        self.pos = 0

    def push(
        self,
        state: torch.Tensor,
        action_indices: List[int],
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Add experience and set its priority to the current max.

        Args:
            state: current state as a torch.Tensor
            action_indices: list of action indices taken in the current state
            reward: reward received for the action taken
            next_state: next state as a torch.Tensor
            done: boolean indicating if the episode has ended

        Returns:
            None

        Raises:
            None
        """
        e = Experience(state, action_indices, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(e)
        else:
            self.buffer[self.pos] = e

        max_prio = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List[Experience], np.ndarray, torch.Tensor]:
        """Sample a batch of experiences from the buffer using prioritized sampling.

        Args:
            batch_size: number of experiences to sample
            beta: importance-sampling exponent (0 = no correction, 1 = full correction)

        Returns:
          experiences: list of Experience
          indices: the positions in the buffer
          weights: importance-sampling weights, as a torch.Tensor

        Raises:
            ValueError: if the buffer is empty
        """

        N = len(self.buffer)
        if N == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        prios = self.priorities[:N] + self.eps
        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(
            weights, dtype=torch.float32, device=experiences[0].state.device
        )

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        """Update the priorities of the experiences at the given indices.

        Args:
            indices: 1-D array of ints, shape (batch_size,)
            errors:  1-D array of floats, shape (batch_size,)

        Returns:
            None

        Raises:
            ValueError: if indices and errors do not have the same length
        """
        if len(indices) != len(errors):
            raise ValueError(
                "Indices and errors must have the same length. "
                f"Got {len(indices)} and {len(errors)}."
            )

        errors = np.array(errors).ravel()

        for idx, err in zip(indices, errors):
            i = int(idx)
            e = float(err)
            self.priorities[i] = abs(e) + self.eps

    def __len__(self) -> int:
        """Return the current size of the buffer.

        Args:
            None

        Returns:
            int: number of experiences in the buffer

        Raises:
            None
        """
        return len(self.buffer)
