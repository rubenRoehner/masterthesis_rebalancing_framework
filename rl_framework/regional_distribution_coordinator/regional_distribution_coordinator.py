"""
regional_distribution_coordinator.py

Defines the RegionalDistributionCoordinator class.
This class implements a multi-head DQN agent with prioritized experience replay for regional e-scooter distribution coordination.
"""

from regional_distribution_coordinator.multi_head_dqn import MultiHeadDQN
from regional_distribution_coordinator.replay_buffer import ReplayBuffer
import torch
from typing import List
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


class RegionalDistributionCoordinator:
    """A multi-head DQN agent for coordinating e-scooter distribution across multiple communities.

    This agent uses separate Q-value heads for each community while sharing feature representations,
    enabling coordinated decision-making across regions with prioritized experience replay.
    """

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
    ) -> None:
        """Initialize the RegionalDistributionCoordinator.

        Args:
            state_dim: dimension of the state space
            num_communities: number of communities/regions to coordinate
            action_values: list of possible action values for vehicle rebalancing
            device: PyTorch device for computations
            replay_buffer_capacity: maximum number of experiences to store
            learning_rate: learning rate for the optimizer
            gamma: discount factor for future rewards
            epsilon_start: initial exploration rate
            epsilon_end: minimum exploration rate
            epsilon_decay: decay rate for epsilon
            batch_size: number of experiences to sample per training step
            tau: soft update parameter for target network
            hidden_dim: dimension of hidden layers in the neural network
            lr_step_size: step size for learning rate scheduler
            lr_gamma: gamma for learning rate scheduler
            replay_buffer_alpha: prioritization exponent for replay buffer
            replay_buffer_beta_start: initial importance sampling exponent
            replay_buffer_beta_frames: frames over which to anneal beta to 1.0

        Returns:
            None

        Raises:
            None
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

        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_buffer_capacity, alpha=self.replay_buffer_alpha
        )

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.learning_rate
        )

        self.scheduler = StepLR(
            self.optimizer,
            step_size=lr_step_size,
            gamma=lr_gamma,
        )

    def select_action(self, state: torch.Tensor) -> List[int]:
        """Select actions for all communities using epsilon-greedy policy.

        Args:
            state: current state of the environment as a torch.Tensor

        Returns:
            List[int]: action indices for each community

        Raises:
            None
        """
        state_tensor = state.unsqueeze(0)

        if torch.rand(1).item() < self.epsilon:
            action_indices = torch.randint(
                low=0, high=self.num_actions_per_head, size=(self.num_communities,)
            )
        else:
            with torch.no_grad():
                q_values_list = self.policy_network(state_tensor)
                action_indices = torch.tensor(
                    [
                        torch.argmax(head_q_values, dim=1).item()
                        for head_q_values in q_values_list
                    ]
                )

        return action_indices.tolist()

    def update_target_network(self) -> None:
        """Hard update of the target network parameters.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def soft_update(self) -> None:
        """Soft update of the target network parameters using tau.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        """
        soft update of the target network parameters
        """
        for targ_param, pol_param in zip(
            self.target_network.parameters(), self.policy_network.parameters()
        ):
            targ_param.data.copy_(
                self.tau * pol_param.data + (1.0 - self.tau) * targ_param.data
            )

    def update_epsilon(self) -> None:
        """Update epsilon using exponential decay.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        """
        Epsilon decay function.
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
    ) -> None:
        """Store an experience in the replay buffer.

        Args:
            state: current state as a torch.Tensor
            action_indices: list of action indices taken
            reward: reward received for the actions
            next_state: next state as a torch.Tensor
            done: boolean indicating if the episode has ended

        Returns:
            None

        Raises:
            None
        """
        self.replay_buffer.push(state, action_indices, reward, next_state, done)

    def anneal_beta(self) -> None:
        """Linearly increase beta from beta_start to 1.0 over beta_frames steps.

        After beta_frames steps, beta stays at 1.0.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        fraction = min(
            float(self.replay_buffer_frames_idx) / self.replay_buffer_beta_frames, 1.0
        )

        self.replay_buffer_beta = self.replay_buffer_beta_start + fraction * (
            1.0 - self.replay_buffer_beta_start
        )

    def train(self) -> tuple[float, torch.Tensor] | None:
        """Train the agent using a batch of experiences from the replay buffer.

        Uses Double DQN with prioritized experience replay and multi-head architecture.

        Args:
            None

        Returns:
            tuple or None: (loss, td_errors) if training occurred, None otherwise

        Raises:
            None
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.replay_buffer_frames_idx += 1
        self.anneal_beta()

        experiences, indices, weights = self.replay_buffer.sample(
            self.batch_size, self.replay_buffer_beta
        )

        if experiences is None:
            return None

        state_batch = torch.stack([exp.state for exp in experiences]).to(self.device)
        action_indice_batch = torch.tensor(
            [exp.action_indices for exp in experiences], device=self.device
        )  # [batch_size, num_communities]
        reward_batch = torch.tensor(
            [exp.reward for exp in experiences], dtype=torch.float32, device=self.device
        )  # [batch_size]

        next_state_batch = torch.stack([exp.next_state for exp in experiences]).to(
            self.device
        )
        done_batch = torch.tensor(
            [exp.done for exp in experiences], dtype=torch.float32, device=self.device
        )  # [batch_size]

        q_values_list = self.policy_network(state_batch)
        current_q_values_selected_list = [
            q_values_list[i].gather(1, action_indice_batch[:, i].unsqueeze(1))
            for i in range(self.num_communities)
        ]  # List of [batch_size, 1] tensors

        with torch.no_grad():
            policy_next_q_values_list = self.policy_network(next_state_batch)
            policy_best_next_actions = [
                q.argmax(dim=1) for q in policy_next_q_values_list
            ]  # [batch_size]

            target_next_q_values_list = self.target_network(next_state_batch)

            evaluated_next_q_values_list = [
                target_next_q_values_list[i]
                .gather(1, policy_best_next_actions[i].unsqueeze(1))
                .squeeze(1)
                for i in range(self.num_communities)
            ]

            stacked_next_max_q_values = torch.stack(
                evaluated_next_q_values_list, dim=1
            )  # [batch_size, num_communities]

            target_q_values = (
                reward_batch.unsqueeze(1)
                + (1 - done_batch.unsqueeze(1)) * self.gamma * stacked_next_max_q_values
            )  # [batch_size, num_communities]

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

        per_sample_losses = per_head_losses.mean(dim=1)  # [batch_size]

        weights = torch.as_tensor(
            weights, dtype=per_head_losses.dtype, device=self.device
        )  # [batch_size]
        weights = weights / weights.max()  # [batch_size]

        loss = (per_sample_losses * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10)
        self.optimizer.step()

        self.scheduler.step()

        self.soft_update()
        self.train_step_counter += 1

        self.update_epsilon()

        return loss.item(), td_per_head.detach()

    def set_evaluation_mode(self, state_dict) -> None:
        """Set the agent to evaluation mode with given state dict.

        Args:
            state_dict: state dictionary to load into networks

        Returns:
            None

        Raises:
            None
        """
        """ """
        self.policy_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)
        self.policy_network.eval()
        self.target_network.eval()
        self.epsilon = 0.0
