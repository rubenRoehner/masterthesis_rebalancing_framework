from datetime import datetime, timedelta
import torch
import pandas as pd
import gymnasium as gym


from demand_forecasting.IrConv_LSTM_demand_forecaster import (
    IrConvLstmDemandForecaster,
)
from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider_impl import DemandProviderImpl
from demand_provider.demand_provider import DemandProvider
from user_incentive_coordinator.escooter_uic_env import EscooterUICEnv
from user_incentive_coordinator.user_incentive_coordinator import (
    UserIncentiveCoordinator,
)

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# global parameters
COMMUNITY_ID = "861faa71fffffff"
FLEET_SIZE = 400
N_EPOCHS = 20
MAX_STEPS_PER_EPISODE = 256
TOTAL_TIME_STEPS = 50_000
START_TIME = datetime(2025, 2, 11, 14, 0)

N_WORKERS = 8
BASE_SEED = 42

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


def make_env(
    rank: int,
    n_zones: int,
    dropoff_demand_forecaster: DemandForecaster,
    pickup_demand_forecaster: DemandForecaster,
    dropoff_demand_provider: DemandProvider,
    pickup_demand_provider: DemandProvider,
    device: torch.device,
    zone_neighbor_map: dict[str, list[str]],
    zone_index_map: dict[str, int],
    seed: int = 0,
):
    def _init():
        env: gym.Env = EscooterUICEnv(
            community_id=COMMUNITY_ID,
            n_zones=n_zones,
            fleet_size=FLEET_SIZE,
            dropoff_demand_forecaster=dropoff_demand_forecaster,
            pickup_demand_forecaster=pickup_demand_forecaster,
            dropoff_demand_provider=dropoff_demand_provider,
            pickup_demand_provider=pickup_demand_provider,
            device=device,
            zone_neighbor_map=zone_neighbor_map,
            zone_index_map=zone_index_map,
            user_willingness=USER_WILLINGNESS,
            max_incentive=MAX_INCENTIVE,
            incentive_levels=INCENTIVE_LEVELS,
            max_steps=MAX_STEPS_PER_EPISODE,
            start_time=START_TIME + timedelta(minutes=rank * UIC_STEP_DURATION),
            step_duration=timedelta(minutes=UIC_STEP_DURATION),
            reward_weight_demand=REWARD_WEIGHT_DEMAND,
            reward_weight_rebalancing=REWARD_WEIGHT_REBALANCING,
            reward_weight_gini=REWARD_WEIGHT_GINI,
        )
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env

    return _init


def main():

    # [grid_index, community_id]
    ZONE_COMMUNITY_MAP: pd.DataFrame = pd.read_pickle(
        "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
    )

    NUM_COMMUNITIES = ZONE_COMMUNITY_MAP["community_index"].nunique()
    N_TOTAL_ZONES = ZONE_COMMUNITY_MAP.shape[0]
    print(f"Total number of communities: {NUM_COMMUNITIES}")
    print(f"Total number of zones: {N_TOTAL_ZONES}")
    print(f"Community ID: {COMMUNITY_ID} ")

    ZONE_INDEX_MAP: dict[str, int] = {}
    for i, row in ZONE_COMMUNITY_MAP[
        ZONE_COMMUNITY_MAP["community_index"] == COMMUNITY_ID
    ].iterrows():
        ZONE_INDEX_MAP.update({row["grid_index"]: i})

    print(f"Number of zones in community {COMMUNITY_ID}: {len(ZONE_INDEX_MAP)}")

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

    # Calculate the number of zones for community COMMUNITY_ID
    N_ZONES = ZONE_COMMUNITY_MAP[
        ZONE_COMMUNITY_MAP["community_index"] == COMMUNITY_ID
    ].shape[0]

    DROP_OFF_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_dropoff_demand_h3_hourly.pickle"
    PICK_UP_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_pickup_demand_h3_hourly.pickle"

    # --- INITIALIZE ENVIRONMENT ---
    dropoff_demand_forecaster = IrConvLstmDemandForecaster(
        num_communities=NUM_COMMUNITIES,
        num_zones=N_TOTAL_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        model_path="/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/models/irregular_convolution_LSTM_dropoff.pkl",
        demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
    )

    pickup_demand_forecaster = IrConvLstmDemandForecaster(
        num_communities=NUM_COMMUNITIES,
        num_zones=N_TOTAL_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        model_path="/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/models/irregular_convolution_LSTM_pickup.pkl",
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

    train_envs = SubprocVecEnv(
        [
            make_env(
                rank=i,
                n_zones=N_ZONES,
                dropoff_demand_forecaster=dropoff_demand_forecaster,
                pickup_demand_forecaster=pickup_demand_forecaster,
                dropoff_demand_provider=dropoff_demand_provider,
                pickup_demand_provider=pickup_demand_provider,
                device=device,
                zone_neighbor_map=ZONE_NEIGHBOR_MAP,
                zone_index_map=ZONE_INDEX_MAP,
                seed=BASE_SEED,
            )
            for i in range(N_WORKERS)
        ]
    )
    train_envs = VecNormalize(train_envs)
    train_envs.save("normalize.pkl")

    eval_env = DummyVecEnv([lambda: Monitor(escooter_env)])
    eval_env = VecNormalize.load("normalize.pkl", eval_env)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=UIC_TENSORBOARD_LOG
        + "/outputs/user_incentive_coordinator/",
        log_path=UIC_TENSORBOARD_LOG + "/outputs/user_incentive_coordinator/eval_logs/",
        eval_freq=1000,
    )

    agent = UserIncentiveCoordinator(
        policy=UIC_POLICY,
        env=train_envs,
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

    train_envs.save(
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
