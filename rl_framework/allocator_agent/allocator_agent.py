from rl_framework.allocator_agent.multi_head_dqn import MultiHeadDQN
from rl_framework.allocator_agent.replay_buffer import ReplayBuffer
import torch


class AllocatorAgent:
    def __init__(self, state_dim, num_communities, action_values, replay_buffer_capacity=10000,
                learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                batch_size=64):
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
            num_heads=num_communities
        )
        self.target_network =  MultiHeadDQN(
            state_dim=state_dim,
            num_actions_per_head=self.num_actions_per_head,
            num_heads=num_communities
        )
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.train_step_counter = 0

        self.replay_buffer_capacity = replay_buffer_capacity
        self.learning_rate = learning_rate

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_capacity)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def select_action(self, state):
        return NotImplementedError("select method not implemented")
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
    
    def store_experience(self, state, action_indices, reward, next_state, done):
        self.replay_buffer.push(state, action_indices, reward, next_state, done)
    
    def train(self):
        return NotImplementedError("train method not implemented")
