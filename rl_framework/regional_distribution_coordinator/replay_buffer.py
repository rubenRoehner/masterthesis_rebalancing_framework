from collections import deque
from typing import NamedTuple, List, Deque, Tuple
import numpy as np
import torch


class Experience(NamedTuple):
    state: torch.Tensor
    action_indices: List[int]
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        """
        capacity: max number of experiences to store
        α: how much prioritization is used (0 = uniform, 1 = full prioritization)
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
        """Add experience and set its priority to the current max."""
        e = Experience(state, action_indices, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(e)
        else:
            # overwrite the oldest
            self.buffer[self.pos] = e

        # set priority of new experience to max so it will be sampled soon
        max_prio = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int, β: float = 0.4
    ) -> Tuple[List[Experience], np.ndarray, torch.Tensor]:
        """
        Returns:
          experiences: list of Experience
          indices: the positions in the buffer
          weights: importance-sampling weights, as a torch.Tensor
        """
        N = len(self.buffer)
        if N == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        # compute sampling probabilities
        prios = self.priorities[:N] + self.eps
        probs = prios**self.alpha
        probs /= probs.sum()

        # draw a batch of indices
        indices = np.random.choice(N, batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        # compute importance-sampling weights
        # w_i = (N * P(i))^{-β} / max_j w_j
        weights = (N * probs[indices]) ** (-β)
        weights /= weights.max()
        weights = torch.tensor(
            weights, dtype=torch.float32, device=experiences[0].state.device
        )

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """
        indices: 1-D array of ints, shape (batch_size,)
        errors:  1-D array of floats, shape (batch_size,)
        """
        # Flatten just in case
        errors = np.array(errors).ravel()

        for idx, err in zip(indices, errors):
            # Convert idx to int, err to float
            i = int(idx)
            e = float(err)
            self.priorities[i] = abs(e) + self.eps

    def __len__(self) -> int:
        return len(self.buffer)
