import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MultiHeadDQN(nn.Module):
    def __init__(self, state_dim, num_actions_per_head, num_heads, hidden_dim=128):
        super(MultiHeadDQN, self).__init__()
        self.num_heads = num_heads
        self.num_actions_per_head = num_actions_per_head

        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Head-specific layers
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions_per_head)
            ) for _ in range(num_heads)
        ])
    
    def forward(self, state):
        # Pass through shared layers
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))

        # Pass through each head and concatenate the outputs
        head_outputs = [head(x) for head in self.heads]
        return torch.cat(head_outputs, dim=1)
