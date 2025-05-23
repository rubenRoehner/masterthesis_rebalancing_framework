from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnv


class UserIncentiveCoordinator:
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
        tensorboard_log: str,
    ):
        policy_kwargs = {
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
            "activation_fn": nn.ReLU,
        }
        self.tensorboard_log = tensorboard_log
        self.model = PPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
        )

    def train(self, total_timesteps: int, callback: MaybeCallback) -> PPO:
        model = self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.model.save(self.tensorboard_log + "/outputs/user_incentive_coordinator")
        print("Model saved as user_incentive_coordinator")
        print("Training complete.")
        return model
