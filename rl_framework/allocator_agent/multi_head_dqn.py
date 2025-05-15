import torch.nn as nn
import torch.nn.functional as F


class MultiHeadDQN(nn.Module):
    def __init__(
        self, state_dim, num_actions_per_head, num_heads, device, hidden_dim=128
    ):
        super(MultiHeadDQN, self).__init__()
        self.num_heads = num_heads
        self.num_actions_per_head = num_actions_per_head
        self.device = device

        # input normalization
        self.input_norm = nn.LayerNorm(state_dim, device=self.device)

        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim, device=self.device)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)

        # Head-specific layers
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

    def forward(self, state):
        # Normalize the input state
        x = self.input_norm(state)

        # Pass through shared layers
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))

        # Pass through each head
        head_outputs = [head(x) for head in self.heads]
        return head_outputs
