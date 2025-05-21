from regional_distribution_coordinator.multi_head_dqn import MultiHeadDQN
from regional_distribution_coordinator.replay_buffer import ReplayBuffer
import torch
from typing import List
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


class RegionalDistributionCoordinator:
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
        tau=0.005,
        hidden_dim=128,
        lr_step_size=1000,
        lr_gamma=0.5,
        replay_buffer_alpha=0.6,
        replay_buffer_beta_start=0.4,
        replay_buffer_beta_frames=50_000,
    ):
        """
        RegionalDistributionCoordinator constructor.
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
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.tau = tau
        self.train_step_counter = 0

        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_buffer_alpha = replay_buffer_alpha

        self.replay_buffer_beta_start = replay_buffer_beta_start
        self.replay_buffer_beta = replay_buffer_beta_start
        self.replay_buffer_beta_frames = replay_buffer_beta_frames
        self.replay_buffer_frames_idx = 0

        # self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_capacity)
        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_buffer_capacity, alpha=self.replay_buffer_alpha
        )

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.learning_rate
        )

        self.scheduler = StepLR(
            self.optimizer,
            step_size=lr_step_size,  # every N steps
            gamma=lr_gamma,  # multiply LR by this
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

    def soft_update(self):
        """
        θ_target ← τ·θ_policy + (1–τ)·θ_target
        """
        for targ_param, pol_param in zip(
            self.target_network.parameters(), self.policy_network.parameters()
        ):
            targ_param.data.copy_(
                self.tau * pol_param.data + (1.0 - self.tau) * targ_param.data
            )

    def update_epsilon(self):
        """
        Update epsilon value slower than the training step counter.
        This is to ensure that the agent explores more in the beginning and
        gradually shifts to exploitation.
        The decay is exponential, so the epsilon value will decrease
        exponentially over time.
        """
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start * (self.epsilon_decay**self.train_step_counter),
        )

    def store_experience(
        self,
        state: torch.Tensor,
        action_indices: List[int],
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        self.replay_buffer.push(state, action_indices, reward, next_state, done)

    def anneal_beta(self):
        """
        Linearly increase beta from beta_start → 1.0 over beta_frames steps.
        After that, beta stays at 1.0.
        """
        fraction = min(
            float(self.replay_buffer_frames_idx) / self.replay_buffer_beta_frames, 1.0
        )
        # beta moves from beta_start up to 1.0
        self.replay_buffer_beta = self.replay_buffer_beta_start + fraction * (
            1.0 - self.replay_buffer_beta_start
        )

    def train(self):
        if len(self.replay_buffer) < self.batch_size:  # not enough samples
            return None

        # Increase the replay buffer index
        # This is used to compute the beta value for importance sampling
        self.replay_buffer_frames_idx += 1
        self.anneal_beta()

        # Sample a batch of experiences from the replay buffer
        experiences, indices, weights = self.replay_buffer.sample(
            self.batch_size, self.replay_buffer_beta
        )

        if experiences is None:
            return None

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
            policy_next_q_values_list = self.policy_network(next_state_batch)
            # For each head, get the max Q-value for the next state. Each element is [batch_size]
            policy_best_next_actions = [
                q.argmax(dim=1) for q in policy_next_q_values_list
            ]  # shape [batch_size]

            target_next_q_values_list = self.target_network(next_state_batch)

            evaluated_next_q_values_list = [
                target_next_q_values_list[i]
                .gather(1, policy_best_next_actions[i].unsqueeze(1))
                .squeeze(1)
                for i in range(self.num_communities)
            ]

            # Stack the evaluated Q-values for each head into a single tensor
            stacked_next_max_q_values = torch.stack(
                evaluated_next_q_values_list, dim=1
            )  # [batch_size, num_communities]

            # Compute the target Q-values
            # target_q_values = reward + (1 - done) * gamma * max_a' Q(s', a')
            target_q_values = (
                reward_batch.unsqueeze(1)
                + (1 - done_batch.unsqueeze(1)) * self.gamma * stacked_next_max_q_values
            )  # [batch_size, num_communities]

        # Compute TD errors for logging and update priorities
        td_per_head = torch.stack(
            [
                (
                    target_q_values[:, i].unsqueeze(1)
                    - current_q_values_selected_list[i]
                ).abs()
                for i in range(self.num_communities)
            ],
            dim=1,
        )  # [batch_size, num_communities]

        td_for_priorities = (
            td_per_head.max(dim=1)[0].detach().cpu().numpy()
        )  # [batch_size]

        self.replay_buffer.update_priorities(indices, td_for_priorities)

        # Compute the loss
        # Used huber loss for stability. MSE yielded unstable training.
        per_head_losses = torch.stack(
            [
                F.huber_loss(
                    current_q_values_selected_list[i],
                    target_q_values[:, i].unsqueeze(1),
                    reduction="none",
                )
                for i in range(self.num_communities)
            ],
            dim=1,
        )  # [batch_size, num_communities]

        # loss per sample => mean over heads/communities
        per_sample_losses = per_head_losses.mean(dim=1)  # [batch_size]

        # normalize weights for stabilizing
        weights = torch.as_tensor(
            weights, dtype=per_head_losses.dtype, device=self.device
        )  # [batch_size]
        weights = weights / weights.max()  # [batch_size]

        loss = (per_sample_losses * weights).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # added gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10)
        self.optimizer.step()

        # Update learning rate
        self.scheduler.step()

        # Update target network
        self.soft_update()
        self.train_step_counter += 1

        # Update epsilon
        self.update_epsilon()

        return loss.item(), td_per_head.detach()
