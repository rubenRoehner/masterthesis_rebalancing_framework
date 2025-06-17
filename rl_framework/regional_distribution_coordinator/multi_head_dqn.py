"""
multi_head_dqn.py

Multi-Head Deep Q-Network implementation.
This module defines a neural network with shared feature extraction and multiple output heads,
designed for multi-action decision making in regional distribution coordination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadDQN(nn.Module):
    """Multi-Head Deep Q-Network with shared feature extraction.

    This network uses shared layers for feature extraction followed by separate heads
    for each community, enabling coordinated but specialized decision-making.
    """

    def __init__(
        self, state_dim, num_actions_per_head, num_heads, device, hidden_dim=128
    ) -> None:
        """Initialize the Multi-Head DQN.

        Args:
            state_dim: dimension of the input state
            num_actions_per_head: number of actions available to each head
            num_heads: number of output heads (communities)
            device: PyTorch device for computations
            hidden_dim: dimension of hidden layers

        Returns:
            None

        Raises:
            None
        """
        super(MultiHeadDQN, self).__init__()
        self.num_heads = num_heads
        self.num_actions_per_head = num_actions_per_head
        self.device = device

        self.input_norm = nn.LayerNorm(state_dim, device=self.device)

        self.shared_fc1 = nn.Linear(state_dim, hidden_dim, device=self.device)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, device=self.device),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_actions_per_head, device=self.device),
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, state) -> list[torch.Tensor]:
        """Forward pass through the network.

        Args:
            state: input state tensor

        Returns:
            list: Q-values for each head as separate tensors

        Raises:
            None
        """
        x = self.input_norm(state)

        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))

        head_outputs = [head(x) for head in self.heads]
        return head_outputs
