from rl_framework.allocator_agent.multi_head_dqn import MultiHeadDQN
from rl_framework.allocator_agent.replay_buffer import ReplayBuffer, Experience
import torch
from typing import List


class AllocatorAgent:
    def __init__(
        self,
        state_dim,
        num_communities,
        action_values,
        replay_buffer_capacity=10000,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=10,
    ):
        """
        AllocatorAgent constructor.
        """
        self.state_dim = state_dim
        self.num_communities = num_communities
        self.action_values = action_values
        self.num_actions_per_head = len(action_values)

        self.policy_network = MultiHeadDQN(
            state_dim=state_dim,
            num_actions_per_head=self.num_actions_per_head,
            num_heads=num_communities,
        )
        self.target_network = MultiHeadDQN(
            state_dim=state_dim,
            num_actions_per_head=self.num_actions_per_head,
            num_heads=num_communities,
        )
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step_counter = 0

        self.replay_buffer_capacity = replay_buffer_capacity
        self.learning_rate = learning_rate

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_capacity)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.learning_rate
        )

    def select_action(self, state: torch.Tensor) -> List[int]:
        # epsilon-greedy action selection

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if torch.rand(1).item() < self.epsilon:
            # Explore: select a random action
            action_indices = torch.randint(
                low=0, high=self.num_actions_per_head, size=(self.num_communities,)
            )
        else:
            # Exploit: select the action with the highest Q-value
            with torch.no_grad():
                self.policy_network.eval()
                q_values = self.policy_network(state_tensor)
                self.policy_network.train()
                action_indices = torch.tensor(
                    [torch.argmax(head, dim=1).item() for head in q_values]
                )

        return action_indices.tolist()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def store_experience(self, state: torch.Tensor, action_indices: List[int], reward: float, next_state: torch.Tensor, done: bool):
        self.replay_buffer.push(state, action_indices, reward, next_state, done)

    def train(self):  # Train the agent using the replay buffer
        if len(self.replay_buffer) < self.batch_size:  # not enough samples
            return None

        # Sample a batch of experiences from the replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)

        state_batch = torch.cat([exp.state for exp in experiences])
        action_indice_batch = torch.tensor([exp.action_indicies for exp in experiences])
        reward_batch = torch.tensor([exp.reward for exp in experiences])
        next_state_batch = torch.cat([exp.next_state for exp in experiences])
        done_batch = torch.tensor([exp.done for exp in experiences])


        # Compute Q-values for the current states
        q_values = self.policy_network(states)

        for i in range(self.num_communities):
            # 

        final_loss = 0

        return final_loss
