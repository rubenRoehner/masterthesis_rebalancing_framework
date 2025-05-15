from datetime import datetime, timedelta
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import pandas as pd

from allocator_agent.allocator_agent import AllocatorAgent
from allocator_agent.escooter_allocator_env import EscooterAllocatorEnv
from demand_forecasting.IrConv_LSTM_demand_forecaster import (
    IrConvLstmDemandForecaster,
)
from demand_provider.demand_provider_impl import DemandProviderImpl


def main():
    # global parameters
    NUM_COMMUNITIES = 8
    NUM_ZONES = 273
    NUM_EPISODES = 100
    MAX_STEPS_PER_EPISODE = 100
    START_TIME = datetime(2025, 2, 11, 14, 0)

    # [grid_id, community_id]
    ZONE_COMMUNITY_MAP: pd.DataFrame = pd.read_pickle(
        "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
    )

    # ALLOCATOR_AGENT parameters
    ALLOCATOR_AGENT_ACTION_VALUES = [-15, -10, -5, 0, 5, 10, 15]
    ALLOCATOR_AGENT_FEATURES_PER_COMMUNITY = (
        3  # forecast for pickup, forecast for dropoff, and current vehicle counts
    )

    ALLOCATOR_AGENT_REPLAY_BUFFER_CAPACITY = 24 * 30  # 24 hours * 30 days buffer
    ALLOCATOR_AGENT_BATCH_SIZE = 32
    ALLOCATOR_AGENT_HIDDEN_DIM = 128

    ALLOCATOR_AGENT_LR = 0.001
    ALLOCATOR_AGENT_GAMMA = 0.99

    ALLOCATOR_AGENT_EPSILON_START = 1.0
    ALLOCATOR_AGENT_EPSILON_END = 0.01
    ALLOCATOR_AGENT_EPSILON_DECAY = 0.9995

    ALLOCATOR_AGENT_STEP_DURATION = 60  # in minutes

    DROP_OFF_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_dropoff_demand_h3_hourly.pickle"
    PICK_UP_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_pickup_demand_h3_hourly.pickle"

    # REBALANCER_AGENT parameters
    # forecast for pickup, forecast for dropoff, and current vehicle counts
    # REBALANCER_AGENT_FEATURES_PER_ZONE = 3

    # --- INITIALIZE ENVIRONMENT ---
    dropoff_demand_forecaster = IrConvLstmDemandForecaster(
        num_communities=NUM_COMMUNITIES,
        num_zones=NUM_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        model_path="/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/models/irregular_convolution_LSTM_37_1747222620_dropoff.pkl",
        demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
    )

    pickup_demand_forecaster = IrConvLstmDemandForecaster(
        num_communities=NUM_COMMUNITIES,
        num_zones=NUM_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        model_path="/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/models/irregular_convolution_LSTM_29_1747224180_pickup.pkl",
        demand_data_path=PICK_UP_DEMAND_DATA_PATH,
    )

    dropoff_demand_provider = DemandProviderImpl(
        num_communities=NUM_COMMUNITIES,
        num_zones=NUM_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
    )

    pickup_demand_provider = DemandProviderImpl(
        num_communities=NUM_COMMUNITIES,
        num_zones=NUM_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        demand_data_path=PICK_UP_DEMAND_DATA_PATH,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    allocator_env = EscooterAllocatorEnv(
        num_communities=NUM_COMMUNITIES,
        features_per_community=ALLOCATOR_AGENT_FEATURES_PER_COMMUNITY,
        action_values=ALLOCATOR_AGENT_ACTION_VALUES,
        max_steps=MAX_STEPS_PER_EPISODE,
        step_duration=timedelta(minutes=ALLOCATOR_AGENT_STEP_DURATION),
        start_time=START_TIME,
        dropoff_demand_forecaster=dropoff_demand_forecaster,
        pickup_demand_forecaster=pickup_demand_forecaster,
        dropoff_demand_provider=dropoff_demand_provider,
        pickup_demand_provider=pickup_demand_provider,
        device=device,
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
        device=device,
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
            "allocator_action_values": ALLOCATOR_AGENT_ACTION_VALUES.__str__(),
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
