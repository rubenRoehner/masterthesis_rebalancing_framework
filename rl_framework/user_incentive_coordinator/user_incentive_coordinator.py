"""
user_incentive_coordinator.py

User Incentive Coordinator using PPO algorithm.
This module implements a reinforcement learning agent that learns to provide optimal incentives
to users for e-scooter rebalancing through behavioral influence.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import get_schedule_fn
import torch


class UserIncentiveCoordinator:
    """A PPO-based agent for coordinating user incentives in e-scooter rebalancing.

    This coordinator learns to provide optimal incentives to influence user behavior,
    encouraging them to drop off scooters in areas where they are most needed.
    """

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: VecEnv,
        learning_rate: float,
        n_steps: int,
        batch_size: int,
        n_epochs: int,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        ent_coef: float,
        verbose: int,
        device: torch.device,
        tensorboard_log: str | None,
        policy_kwargs: dict | None = None,
        vf_coef: float = 0.5,
        target_kl: float | None = None,
    ) -> None:
        """Initialize the User Incentive Coordinator.

        Args:
            policy: policy class or string identifier for the PPO policy
            env: vectorized environment for training
            learning_rate: learning rate for the optimizer
            n_steps: number of steps to run for each environment per update
            batch_size: minibatch size for training
            n_epochs: number of epochs to train on each batch
            gamma: discount factor for future rewards
            gae_lambda: factor for trade-off of bias vs variance for GAE
            clip_range: clipping parameter for PPO
            ent_coef: entropy coefficient for regularization
            verbose: verbosity level for training output
            tensorboard_log: path for tensorboard logging
            policy_kwargs: additional arguments for policy network architecture
            vf_coef: value function coefficient in the loss calculation
            target_kl: target KL divergence threshold for early stopping

        Returns:
            None

        Raises:
            None
        """

        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": dict(pi=[64, 64], vf=[64, 64]),
                "activation_fn": nn.ReLU,
            }
        self.tensorboard_log = tensorboard_log

        self.model = PPO(
            policy=policy,
            env=env,
            learning_rate=get_schedule_fn(learning_rate),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=get_schedule_fn(clip_range),
            ent_coef=ent_coef,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            vf_coef=vf_coef,
            target_kl=target_kl,
            device=device,
        )

    def train(self, total_timesteps: int, callback: MaybeCallback) -> PPO:
        """Train the User Incentive Coordinator using PPO algorithm.

        Args:
            total_timesteps: total number of timesteps to train for
            callback: callback function(s) called during training

        Returns:
            PPO: the trained PPO model

        Raises:
            None
        """
        model = self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            tb_log_name="UserIncentiveCoordinator",
        )
        if self.tensorboard_log:
            model.save(
                self.tensorboard_log + "/outputs/user_incentive_coordinator/model"
            )
            print("Model saved in tensorboard log directory.")
        print("Training complete.")
        return model
