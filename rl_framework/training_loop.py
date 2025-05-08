from datetime import datetime, timedelta
from torch.utils.tensorboard.writer import SummaryWriter
import torch

from rl_framework.allocator_agent.allocator_agent import AllocatorAgent
from rl_framework.allocator_agent.escooter_allocator_env import EscooterAllocatorEnv


def main():
    # global parameters
    NUM_COMMUNITIES = 8
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 100
    START_TIME = datetime(2025, 2, 11, 14, 0)

    # ALLOCATOR_AGENT parameters
    ALLOCATOR_AGENT_ACTION_VALUES = [-15, -10, -5, 0, 5, 10, 15]
    ALLOCATOR_AGENT_FEATURES_PER_COMMUNITY = (
        3  # forecast for pickup, forecast for dropoff, and current vehicle counts
    )

    ALLOCATOR_AGENT_REPLAY_BUFFER_CAPACITY = 10000
    ALLOCATOR_AGENT_BATCH_SIZE = 32
    ALLOCATOR_AGENT_HIDDEN_DIM = 128

    ALLOCATOR_AGENT_LR = 0.001
    ALLOCATOR_AGENT_GAMMA = 0.99

    ALLOCATOR_AGENT_EPSILON_START = 1.0
    ALLOCATOR_AGENT_EPSILON_END = 0.01
    ALLOCATOR_AGENT_EPSILON_DECAY = 0.995

    ALLOCATOR_AGENT_STEP_DURATION = 60  # in minutes

    # REBALANCER_AGENT parameters
    # forecast for pickup, forecast for dropoff, and current vehicle counts
    # REBALANCER_AGENT_FEATURES_PER_ZONE = 3

    # --- INITIALIZE ENVIRONMENT ---

    allocator_env = EscooterAllocatorEnv(
        num_communities=NUM_COMMUNITIES,
        features_per_community=ALLOCATOR_AGENT_FEATURES_PER_COMMUNITY,
        action_values=ALLOCATOR_AGENT_ACTION_VALUES,
        max_steps=MAX_STEPS_PER_EPISODE,
        step_duration=timedelta(minutes=ALLOCATOR_AGENT_STEP_DURATION),
        start_time=START_TIME,
    )

    allocator_agent = AllocatorAgent(
        state_dim=NUM_COMMUNITIES * ALLOCATOR_AGENT_FEATURES_PER_COMMUNITY,
        num_communities=NUM_COMMUNITIES,
        action_values=ALLOCATOR_AGENT_ACTION_VALUES,
        replay_buffer_capacity=ALLOCATOR_AGENT_REPLAY_BUFFER_CAPACITY,
        learning_rate=ALLOCATOR_AGENT_LR,
        gamma=ALLOCATOR_AGENT_GAMMA,
        epsilon_start=ALLOCATOR_AGENT_EPSILON_START,
        epsilon_end=ALLOCATOR_AGENT_EPSILON_END,
        epsilon_decay=ALLOCATOR_AGENT_EPSILON_DECAY,
        batch_size=ALLOCATOR_AGENT_BATCH_SIZE,
        hidden_dim=ALLOCATOR_AGENT_HIDDEN_DIM,
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(
        f"runs/allocator_agent_experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    writer.add_hparams(
        {
            "num_communities": NUM_COMMUNITIES,
            "num_episodes": NUM_EPISODES,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "allocator_step_duration": ALLOCATOR_AGENT_STEP_DURATION,
            "allocator_epsilon_start": ALLOCATOR_AGENT_EPSILON_START,
            "allocator_epsilon_end": ALLOCATOR_AGENT_EPSILON_END,
            "allocator_epsilon_decay": ALLOCATOR_AGENT_EPSILON_DECAY,
            "allocator_learning_rate": ALLOCATOR_AGENT_LR,
            "allocator_gamma": ALLOCATOR_AGENT_GAMMA,
            "allocator_batch_size": ALLOCATOR_AGENT_BATCH_SIZE,
            "allocator_replay_buffer_capacity": ALLOCATOR_AGENT_REPLAY_BUFFER_CAPACITY,
            "allocator_action_values": ALLOCATOR_AGENT_ACTION_VALUES,
            "allocator_features_per_community": ALLOCATOR_AGENT_FEATURES_PER_COMMUNITY,
        },
        {},
    )

    for episode in range(NUM_EPISODES):
        current_observation, info = allocator_env.reset()
        total_episode_reward = 0
        episode_loss = 0
        num_training_steps = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            # Get action from the allocator agent
            action = allocator_agent.select_action(current_observation)

            # Take action in the environment
            next_observation, reward, terminated, truncated, info = allocator_env.step(
                action
            )
            done = terminated or truncated

            allocator_agent.store_experience(
                current_observation, action, reward, next_observation, done
            )

            loss = allocator_agent.train()
            if loss is not None:
                episode_loss += loss
                num_training_steps += 1

            current_observation = next_observation
            total_episode_reward += reward

            if done:
                break

        # Log episode metrics
        writer.add_scalar("Reward/Episode", total_episode_reward, episode)
        if num_training_steps > 0:
            writer.add_scalar(
                "Loss/Episode_Avg", episode_loss / num_training_steps, episode
            )
        writer.add_scalar(
            "Epsilon/Episode", allocator_agent.epsilon, episode
        )  # Log current epsilon

        print(
            f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_episode_reward:.2f}, Epsilon: {allocator_agent.epsilon:.2f}"
        )
        if num_training_steps > 0:
            print(f"Average Loss: {episode_loss / num_training_steps:.4f}")

    torch.save(
        allocator_agent.policy_network.state_dict(),
        f"allocator_agent_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth",
    )
    allocator_env.close()
    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
