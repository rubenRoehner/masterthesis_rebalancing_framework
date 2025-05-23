from datetime import datetime, timedelta
import torch
import pandas as pd

from demand_forecasting.IrConv_LSTM_demand_forecaster import (
    IrConvLstmDemandForecaster,
)
from demand_provider.demand_provider_impl import DemandProviderImpl
from user_incentive_coordinator.escooter_uic_env import EscooterUICEnv
from user_incentive_coordinator.user_incentive_coordinator import (
    UserIncentiveCoordinator,
)

from stable_baselines3.common.vec_env import (
    VecNormalize,
    DummyVecEnv,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


def main():
    # global parameters
    NUM_COMMUNITIES = 8
    COMMUNITY_ID = "861faa71fffffff"
    N_TOTAL_ZONES = 273
    FLEET_SIZE = 400
    N_EPOCHS = 20
    MAX_STEPS_PER_EPISODE = 256
    TOTAL_TIME_STEPS = 200_000
    START_TIME = datetime(2025, 2, 11, 14, 0)

    # [grid_index, community_id]
    ZONE_COMMUNITY_MAP: pd.DataFrame = pd.read_pickle(
        "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
    )

    ZONE_INDEX_MAP: dict[str, int] = {}
    for i, row in ZONE_COMMUNITY_MAP[
        ZONE_COMMUNITY_MAP["community_index"] == COMMUNITY_ID
    ].iterrows():
        ZONE_INDEX_MAP.update({row["grid_index"]: i})

    COMMUNTIY_ZONE_IDS = set(ZONE_INDEX_MAP.keys())

    ZONE_NEIGHBOR_MAP_DF: pd.DataFrame = pd.read_pickle(
        "/home/ruroit00/rebalancing_framework/processed_data/grid_cells_neighbors_list.pickle"
    )  # [zone_index, list of neighbors]

    ZONE_NEIGHBOR_MAP: dict[str, list[str]] = {}
    for i, row in ZONE_NEIGHBOR_MAP_DF.iterrows():
        if row["grid_index"] not in COMMUNTIY_ZONE_IDS:
            continue
        # Filter neighbors to only include those in the same community
        neighbors = [
            neighbor for neighbor in row["neighbors"] if neighbor in COMMUNTIY_ZONE_IDS
        ]
        ZONE_NEIGHBOR_MAP.update({row["grid_index"]: neighbors})

    # Calculate the number of zones fore community COMMUNITY_ID
    N_ZONES = ZONE_COMMUNITY_MAP[
        ZONE_COMMUNITY_MAP["community_index"] == COMMUNITY_ID
    ].shape[0]

    DROP_OFF_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_dropoff_demand_h3_hourly.pickle"
    PICK_UP_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_pickup_demand_h3_hourly.pickle"

    # UIC parameters
    UIC_STEP_DURATION = 60  # in minutes

    USER_WILLINGNESS = [0.0, 0.05, 0.1, 0.15, 0.3]
    MAX_INCENTIVE = 5.0
    INCENTIVE_LEVELS = 5

    REWARD_WEIGHT_DEMAND = 1.0
    REWARD_WEIGHT_REBALANCING = 0.5
    REWARD_WEIGHT_GINI = 0.25

    UIC_POLICY = "MultiInputPolicy"
    UIC_N_STEPS = 256
    UIC_LEARNING_RATE = 3e-4
    UIC_GAMMA = 0.99
    UIC_GAE_LAMBDA = 0.95
    UIC_CLIP_RANGE = 0.2
    UIC_ENT_COEF = 0.01
    UIC_BATCH_SIZE = 32
    UIC_VERBOSE = 1
    UIC_TENSORBOARD_LOG = "rl_framework/runs/"

    # --- INITIALIZE ENVIRONMENT ---
    dropoff_demand_forecaster = IrConvLstmDemandForecaster(
        num_communities=NUM_COMMUNITIES,
        num_zones=N_TOTAL_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        model_path="/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/models/irregular_convolution_LSTM_37_1747222620_dropoff.pkl",
        demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
    )

    pickup_demand_forecaster = IrConvLstmDemandForecaster(
        num_communities=NUM_COMMUNITIES,
        num_zones=N_TOTAL_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        model_path="/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/models/irregular_convolution_LSTM_29_1747224180_pickup.pkl",
        demand_data_path=PICK_UP_DEMAND_DATA_PATH,
    )

    dropoff_demand_provider = DemandProviderImpl(
        num_communities=NUM_COMMUNITIES,
        num_zones=N_TOTAL_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
    )

    pickup_demand_provider = DemandProviderImpl(
        num_communities=NUM_COMMUNITIES,
        num_zones=N_TOTAL_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        demand_data_path=PICK_UP_DEMAND_DATA_PATH,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    escooter_env = EscooterUICEnv(
        community_id=COMMUNITY_ID,
        n_zones=N_ZONES,
        fleet_size=FLEET_SIZE,
        dropoff_demand_forecaster=dropoff_demand_forecaster,
        pickup_demand_forecaster=pickup_demand_forecaster,
        dropoff_demand_provider=dropoff_demand_provider,
        pickup_demand_provider=pickup_demand_provider,
        device=device,
        zone_neighbor_map=ZONE_NEIGHBOR_MAP,
        zone_index_map=ZONE_INDEX_MAP,
        user_willingness=USER_WILLINGNESS,
        max_incentive=MAX_INCENTIVE,
        incentive_levels=INCENTIVE_LEVELS,
        max_steps=MAX_STEPS_PER_EPISODE,
        start_time=START_TIME,
        step_duration=timedelta(minutes=UIC_STEP_DURATION),
        reward_weight_demand=REWARD_WEIGHT_DEMAND,
        reward_weight_rebalancing=REWARD_WEIGHT_REBALANCING,
        reward_weight_gini=REWARD_WEIGHT_GINI,
    )

    env = DummyVecEnv([lambda: Monitor(escooter_env)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    eval_env = DummyVecEnv([lambda: Monitor(escooter_env)])
    # eval_env VecNormalize.load("normalize.pkl", eval_env)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=UIC_TENSORBOARD_LOG
        + "/outputs/user_incentive_coordinator/",
        log_path=UIC_TENSORBOARD_LOG + "/outputs/user_incentive_coordinator/eval_logs/",
        eval_freq=10_000,
    )

    agent = UserIncentiveCoordinator(
        policy=UIC_POLICY,
        env=env,
        learning_rate=UIC_LEARNING_RATE,
        n_steps=UIC_N_STEPS,
        n_epochs=N_EPOCHS,
        batch_size=UIC_BATCH_SIZE,
        gamma=UIC_GAMMA,
        gae_lambda=UIC_GAE_LAMBDA,
        clip_range=UIC_CLIP_RANGE,
        ent_coef=UIC_ENT_COEF,
        verbose=UIC_VERBOSE,
        tensorboard_log=UIC_TENSORBOARD_LOG,
    )

    agent.train(
        total_timesteps=TOTAL_TIME_STEPS,
        callback=eval_callback,
    )

    env.save(
        UIC_TENSORBOARD_LOG
        + "/outputs/user_incentive_coordinator/"
        + COMMUNITY_ID
        + "_env_train_normalize.pkl"
    )

    agent.model.save(
        UIC_TENSORBOARD_LOG
        + "/outputs/user_incentive_coordinator/"
        + COMMUNITY_ID
        + "_user_incentive_coordinator"
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
