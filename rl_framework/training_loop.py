from datetime import datetime, timedelta
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import pandas as pd
import numpy as np

from regional_distribution_coordinator.regional_distribution_coordinator import (
    RegionalDistributionCoordinator,
)
from regional_distribution_coordinator.escooter_rdc_env import EscooterRDCEnv
from demand_forecasting.IrConv_LSTM_demand_forecaster import (
    IrConvLstmDemandForecaster,
)
from demand_provider.demand_provider_impl import DemandProviderImpl


def main():
    # global parameters
    NUM_COMMUNITIES = 8
    NUM_ZONES = 273
    FLEET_SIZE = 1000
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 100
    START_TIME = datetime(2025, 2, 11, 14, 0)

    # [grid_id, community_id]
    ZONE_COMMUNITY_MAP: pd.DataFrame = pd.read_pickle(
        "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
    )

    # RegionalDistributionCoordinator parameters
    RDC_ACTION_VALUES = [-15, -10, -5, 0, 5, 10, 15]
    RDC_FEATURES_PER_COMMUNITY = (
        3  # forecast for pickup, forecast for dropoff, and current vehicle counts
    )

    RDC_REPLAY_BUFFER_CAPACITY = 10000
    RDC_REPLAY_BUFFER_ALPHA = 0.6
    RDC_REPLAY_BUFFER_BETA_START = 0.4
    RDC_REPLAY_BUFFER_BETA_FRAMES = 100_000

    RDC_BATCH_SIZE = 256
    RDC_HIDDEN_DIM = 128

    RDC_TARGET_UPDATE_FREQ = 1000
    RDC_LR = 5e-6
    RDC_LR_STEP_SIZE = 1000
    RDC_LR_GAMMA = 0.5
    RDC_GAMMA = 0.99

    RDC_EPSILON_START = 1.0
    RDC_EPSILON_END = 0.05
    RDC_EPSILON_DECAY = 0.999

    RDC_TAU = 0.001

    RDC_STEP_DURATION = 60  # in minutes

    RDC_OPERATOR_REBALANCING_COST = 0.5
    RDC_REWARD_WEIGHT_DEMAND = 1.0
    RDC_REWARD_WEIGHT_REBALANCING = 1.0
    RDC_REWARD_WEIGHT_GINI = 0.25

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

    rdc_env = EscooterRDCEnv(
        num_communities=NUM_COMMUNITIES,
        features_per_community=RDC_FEATURES_PER_COMMUNITY,
        action_values=RDC_ACTION_VALUES,
        max_steps=MAX_STEPS_PER_EPISODE,
        step_duration=timedelta(minutes=RDC_STEP_DURATION),
        start_time=START_TIME,
        operator_rebalancing_cost=RDC_OPERATOR_REBALANCING_COST,
        reward_weight_demand=RDC_REWARD_WEIGHT_DEMAND,
        reward_weight_rebalancing=RDC_REWARD_WEIGHT_REBALANCING,
        reward_weight_gini=RDC_REWARD_WEIGHT_GINI,
        dropoff_demand_forecaster=dropoff_demand_forecaster,
        pickup_demand_forecaster=pickup_demand_forecaster,
        dropoff_demand_provider=dropoff_demand_provider,
        pickup_demand_provider=pickup_demand_provider,
        device=device,
        fleet_size=FLEET_SIZE,
    )

    rdc_agent = RegionalDistributionCoordinator(
        state_dim=NUM_COMMUNITIES * RDC_FEATURES_PER_COMMUNITY,
        num_communities=NUM_COMMUNITIES,
        action_values=RDC_ACTION_VALUES,
        replay_buffer_capacity=RDC_REPLAY_BUFFER_CAPACITY,
        replay_buffer_alpha=RDC_REPLAY_BUFFER_ALPHA,
        replay_buffer_beta_start=RDC_REPLAY_BUFFER_BETA_START,
        replay_buffer_beta_frames=RDC_REPLAY_BUFFER_BETA_FRAMES,
        learning_rate=RDC_LR,
        lr_step_size=RDC_LR_STEP_SIZE,
        lr_gamma=RDC_LR_GAMMA,
        gamma=RDC_GAMMA,
        epsilon_start=RDC_EPSILON_START,
        epsilon_end=RDC_EPSILON_END,
        epsilon_decay=RDC_EPSILON_DECAY,
        batch_size=RDC_BATCH_SIZE,
        hidden_dim=RDC_HIDDEN_DIM,
        target_update_freq=RDC_TARGET_UPDATE_FREQ,
        tau=RDC_TAU,
        device=device,
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(
        f"rl_framework/runs/regional_distribution_coordinator_experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    writer.add_hparams(
        {
            "num_communities": NUM_COMMUNITIES,
            "num_episodes": NUM_EPISODES,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "rdc_step_duration": RDC_STEP_DURATION,
            "rdc_epsilon_start": RDC_EPSILON_START,
            "rdc_epsilon_end": RDC_EPSILON_END,
            "rdc_epsilon_decay": RDC_EPSILON_DECAY,
            "rdc_learning_rate": RDC_LR,
            "rdc_lr_step_size": RDC_LR_STEP_SIZE,
            "rdc_lr_gamma": RDC_LR_GAMMA,
            "rdc_gamma": RDC_GAMMA,
            "rdc_batch_size": RDC_BATCH_SIZE,
            "rdc_replay_buffer_capacity": RDC_REPLAY_BUFFER_CAPACITY,
            "rdc_replay_buffer_alpha": RDC_REPLAY_BUFFER_ALPHA,
            "rdc_replay_buffer_beta_start": RDC_REPLAY_BUFFER_BETA_START,
            "rdc_replay_buffer_beta_frames": RDC_REPLAY_BUFFER_BETA_FRAMES,
            "rdc_action_values": RDC_ACTION_VALUES.__str__(),
            "rdc_features_per_community": RDC_FEATURES_PER_COMMUNITY,
            "rdc_hidden_dim": RDC_HIDDEN_DIM,
            "rdc_target_update_freq": RDC_TARGET_UPDATE_FREQ,
            "rdc_tau": RDC_TAU,
            "rdc_operator_rebalancing_cost": RDC_OPERATOR_REBALANCING_COST,
            "rdc_reward_weight_demand": RDC_REWARD_WEIGHT_DEMAND,
            "rdc_reward_weight_rebalancing": RDC_REWARD_WEIGHT_REBALANCING,
            "rdc_reward_weight_gini": RDC_REWARD_WEIGHT_GINI,
        },
        {},
    )

    global_step = 0

    for episode in range(NUM_EPISODES):
        current_observation, info = rdc_env.reset()
        total_episode_reward = 0
        episode_loss = 0
        num_training_steps = 0

        episode_vehicles_rebalanced = 0
        actions_this_episode = []

        for step in range(MAX_STEPS_PER_EPISODE):
            # Get action from the rdc agent
            action = rdc_agent.select_action(current_observation)
            actions_this_episode.append(action)

            # Take action in the environment
            next_observation, reward, terminated, truncated, info = rdc_env.step(action)
            done = terminated or truncated

            rdc_agent.store_experience(
                current_observation, action, reward, next_observation, done
            )

            output = rdc_agent.train()
            if output is not None:
                loss, td_errors = output
                episode_loss += loss
                num_training_steps += 1

                writer.add_scalar("TD_Error/Mean", td_errors.abs().mean(), global_step)
                writer.add_scalar("TD_Error/Max", td_errors.abs().max(), global_step)

            current_observation = next_observation
            total_episode_reward += reward

            episode_vehicles_rebalanced += info["total_vehicles_rebalanced"]
            global_step += 1
            if done:
                break

        # Log episode metrics
        writer.add_scalar("Reward/Episode", total_episode_reward, episode)
        if num_training_steps > 0:
            writer.add_scalar(
                "Loss/Episode_Avg", episode_loss / num_training_steps, episode
            )
        # Log current epsilon
        writer.add_scalar("Epsilon/Episode", rdc_agent.epsilon, episode)

        # Log  vehicles rebalanced
        writer.add_scalar(
            "VehiclesRebalanced/Episode", episode_vehicles_rebalanced, episode
        )

        # Log actions per head as histogram
        actions_array = np.array(actions_this_episode, dtype=int)
        for head in range(NUM_COMMUNITIES):
            writer.add_histogram(
                f"Action/Head{head}", actions_array[:, head], global_step=episode
            )

        # log Buffer size
        writer.add_scalar("Buffer/Size", len(rdc_agent.replay_buffer), episode)

        print(
            f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_episode_reward:.2f}, Epsilon: {rdc_agent.epsilon:.2f}"
        )
        if num_training_steps > 0:
            print(f"Average Loss: {episode_loss / num_training_steps:.4f}")

    torch.save(
        rdc_agent.policy_network.state_dict(),
        f"rl_framework/runs/outputs/rdc_agent_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth",
    )
    rdc_env.close()
    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
