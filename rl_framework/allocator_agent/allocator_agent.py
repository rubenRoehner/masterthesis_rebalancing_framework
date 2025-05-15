from allocator_agent.multi_head_dqn import MultiHeadDQN
from allocator_agent.replay_buffer import ReplayBuffer
import torch
from typing import List
import torch.nn.functional as F


class AllocatorAgent:
    def __init__(
        self,
        state_dim,
        num_communities,
        action_values,
        device: torch.device,
        replay_buffer_capacity=10000,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        batch_size=64,
        target_update_freq=10,
        hidden_dim=128,
    ):
        """
        AllocatorAgent constructor.
        """
        self.state_dim = state_dim
        self.num_communities = num_communities
        self.action_values = action_values
        self.num_actions_per_head = len(action_values)

        self.device = device

        self.policy_network = MultiHeadDQN(
            state_dim=state_dim,
            num_actions_per_head=self.num_actions_per_head,
            num_heads=num_communities,
            hidden_dim=hidden_dim,
            device=self.device,
        )
        self.target_network = MultiHeadDQN(
            state_dim=state_dim,
            num_actions_per_head=self.num_actions_per_head,
            num_heads=num_communities,
            hidden_dim=hidden_dim,
            device=self.device,
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
        """
        Select an action based on the current state using epsilon-greedy policy
        :param state: The current state of the environment (tensor)
        :return: The selected action index for each community
        """
        # epsilon-greedy action selection

        # state is already a tensor from the environment
        state_tensor = state.unsqueeze(0)  # Add batch dimension

        if torch.rand(1).item() < self.epsilon:
            # Explore: select a random action
            action_indices = torch.randint(
                low=0, high=self.num_actions_per_head, size=(self.num_communities,)
            )
        else:
            # Exploit: select the action with the highest Q-value
            with torch.no_grad():
                q_values_list = self.policy_network(state_tensor)
                action_indices = torch.tensor(
                    [
                        torch.argmax(head_q_values, dim=1).item()
                        for head_q_values in q_values_list
                    ]
                )

        return action_indices.tolist()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def store_experience(
        self,
        state: torch.Tensor,
        action_indices: List[int],
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        self.replay_buffer.push(state, action_indices, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:  # not enough samples
            return None

        # Sample a batch of experiences from the replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)

        # exp.state is a 1D tensor [state_dim]. stack creates [batch_size, state_dim]
        state_batch = torch.stack([exp.state for exp in experiences]).to(self.device)
        action_indice_batch = torch.tensor(
            [exp.action_indices for exp in experiences], device=self.device
        )  # [batch_size, num_communities]
        reward_batch = torch.tensor(
            [exp.reward for exp in experiences], dtype=torch.float32, device=self.device
        )  # [batch_size]

        # exp.next_state is a 1D tensor [state_dim]. stack creates [batch_size, state_dim]
        next_state_batch = torch.stack([exp.next_state for exp in experiences]).to(
            self.device
        )
        done_batch = torch.tensor(
            [exp.done for exp in experiences], dtype=torch.float32, device=self.device
        )  # [batch_size]

        # Compute Q-values for the current states
        # policy_network returns a list of tensors, one for each head e.g. for each communtiy.
        # Each tensor in q_values_list is [batch_size, num_actions_per_head]
        q_values_list = self.policy_network(state_batch)
        current_q_values_selected_list = [
            q_values_list[i].gather(1, action_indice_batch[:, i].unsqueeze(1))
            for i in range(self.num_communities)
        ]  # List of [batch_size, 1] tensors

        with torch.no_grad():
            # Compute Q-values for the next states using the target network
            # target_network also returns a list of tensors
            next_q_values_list = self.target_network(next_state_batch)
            # For each head, get the max Q-value for the next state. Each element is [batch_size]
            next_max_q_values_per_head_list = [
                next_q_values_list[i].max(1)[0] for i in range(self.num_communities)
            ]
            # Stack them to get [batch_size, num_communities]
            stacked_next_max_q_values = torch.stack(
                next_max_q_values_per_head_list, dim=1
            )

            # Compute the target Q-values: R + gamma * max_a' Q_target(s', a')
            # reward_batch is [batch_size], done_batch is [batch_size]
            # stacked_next_max_q_values is [batch_size, num_communities]
            # Broadcasting reward_batch and done_batch for element-wise multiplication
            target_q_values = (
                reward_batch.unsqueeze(1)  # [batch_size, 1]
                + (1 - done_batch.unsqueeze(1))  # [batch_size, 1]
                * self.gamma
                * stacked_next_max_q_values  # [batch_size, num_communities]
            )  # Result is [batch_size, num_communities]

        # Compute TD errors for logging
        td_errors_per_head = [
            (target_q_values[:, i].unsqueeze(1) - current_q_values_selected_list[i])
            for i in range(self.num_communities)
        ]
        td_errors = torch.cat(td_errors_per_head, dim=1).detach()

        # Compute the loss
        per_head_losses = [
            F.huber_loss(
                current_q_values_selected_list[i],
                target_q_values[:, i].unsqueeze(1),
                reduction="mean",
            )
            for i in range(self.num_communities)
        ]
        loss = torch.stack(per_head_losses).to(self.device).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10)
        self.optimizer.step()

        self.train_step_counter += 1

        # Update the target network every target_update_freq steps
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        # Update epsilon
        self.update_epsilon()

        return loss.item(), td_errors
