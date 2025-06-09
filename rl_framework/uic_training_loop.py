from datetime import datetime, timedelta
import torch
import pandas as pd
import gymnasium as gym


from demand_forecasting.IrConv_LSTM_pre_forecaster import (
    IrConvLstmDemandPreForecaster,
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

# Best hyperparameters: {'learning_rate': 1.0920802054925035e-06, 'n_steps': 512, 'batch_size': 32, 'gamma': 0.918, 'clip_range': 0.13, 'ent_coef': 0.0007835438398052651, 'vf_coef': 0.6, 'use_target_kl': 0.02, 'n_layers': 3, 'hidden_size': 256, 'activation': 'Tanh'}
# global parameters
COMMUNITY_IDS = [
    "861faa44fffffff",
    "861faa637ffffff",
    "861faa707ffffff",
    "861faa717ffffff",
    "861faa71fffffff",
    "861faa787ffffff",
    "861faa78fffffff",
    "861faa7a7ffffff",
    "861faa7afffffff",
]
FLEET_SIZE = 40
N_EPOCHS = 20
MAX_STEPS_PER_EPISODE = 256
TOTAL_TIME_STEPS = 200_000
START_TIME = datetime(2025, 2, 11, 14, 0)

torch.cuda.set_device(2)
N_WORKERS = 8
BASE_SEED = 42

# UIC parameters
UIC_STEP_DURATION = 60  # in minutes

REWARD_WEIGHT_DEMAND = 1.0
REWARD_WEIGHT_REBALANCING = 0.5
REWARD_WEIGHT_GINI = 0.25

UIC_POLICY = "MultiInputPolicy"
UIC_N_STEPS = 512
UIC_LEARNING_RATE = 1.092e-06
UIC_GAMMA = 0.918
UIC_GAE_LAMBDA = 0.95
UIC_CLIP_RANGE = 0.13
UIC_ENT_COEF = 0.00078
UIC_BATCH_SIZE = 32
UIC_VERBOSE = 1
UIC_POLICY_KWARGS = {
    "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
    "activation_fn": torch.nn.Tanh,
}
UIC_VF_COEF = 0.6
UIC_TARGET_KL = 0.02
UIC_TENSORBOARD_LOG = "rl_framework/runs/UIC/"


@staticmethod
def USER_WILLINGNESS_FN(incentive: float) -> float:
    """ """
    return 0.4 * incentive


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
    community_id: str,
    seed: int = 0,
):
    def _init():
        env: gym.Env = EscooterUICEnv(
            community_id=community_id,
            n_zones=n_zones,
            fleet_size=FLEET_SIZE,
            dropoff_demand_forecaster=dropoff_demand_forecaster,
            pickup_demand_forecaster=pickup_demand_forecaster,
            dropoff_demand_provider=dropoff_demand_provider,
            pickup_demand_provider=pickup_demand_provider,
            device=device,
            zone_neighbor_map=zone_neighbor_map,
            zone_index_map=zone_index_map,
            user_willingness_fn=USER_WILLINGNESS_FN,
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


def train_uic(community_id: str):

    # [grid_index, community_id]
    ZONE_COMMUNITY_MAP: pd.DataFrame = pd.read_pickle(
        "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
    )

    NUM_COMMUNITIES = ZONE_COMMUNITY_MAP["community_index"].nunique()
    N_TOTAL_ZONES = ZONE_COMMUNITY_MAP.shape[0]
    print(f"Communities: {ZONE_COMMUNITY_MAP['community_index'].unique()}")
    print(f"Total number of zones: {N_TOTAL_ZONES}")
    print(f"Community ID: {community_id} ")

    ZONE_INDEX_MAP: dict[str, int] = {}
    for i, row in ZONE_COMMUNITY_MAP[
        ZONE_COMMUNITY_MAP["community_index"] == community_id
    ].iterrows():
        ZONE_INDEX_MAP.update({row["grid_index"]: i})

    print(f"Number of zones in community {community_id}: {len(ZONE_INDEX_MAP)}")

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
        ZONE_COMMUNITY_MAP["community_index"] == community_id
    ].shape[0]

    DROP_OFF_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_dropoff_demand_h3_hourly.pickle"
    PICK_UP_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_pickup_demand_h3_hourly.pickle"

    DROP_OFF_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_dropoff_forecasts.pkl"
    PICK_UP_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_pickup_forecasts.pkl"

    # --- INITIALIZE ENVIRONMENT ---
    dropoff_demand_forecaster = IrConvLstmDemandPreForecaster(
        num_communities=NUM_COMMUNITIES,
        num_zones=N_TOTAL_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        demand_data_path=DROP_OFF_DEMAND_FORECAST_DATA_PATH,
    )

    pickup_demand_forecaster = IrConvLstmDemandPreForecaster(
        num_communities=NUM_COMMUNITIES,
        num_zones=N_TOTAL_ZONES,
        zone_community_map=ZONE_COMMUNITY_MAP,
        demand_data_path=PICK_UP_DEMAND_FORECAST_DATA_PATH,
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
        community_id=community_id,
        n_zones=N_ZONES,
        fleet_size=FLEET_SIZE,
        dropoff_demand_forecaster=dropoff_demand_forecaster,
        pickup_demand_forecaster=pickup_demand_forecaster,
        dropoff_demand_provider=dropoff_demand_provider,
        pickup_demand_provider=pickup_demand_provider,
        device=device,
        zone_neighbor_map=ZONE_NEIGHBOR_MAP,
        zone_index_map=ZONE_INDEX_MAP,
        user_willingness_fn=USER_WILLINGNESS_FN,
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
                community_id=community_id,
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
        best_model_save_path=UIC_TENSORBOARD_LOG + "/outputs/",
        log_path=UIC_TENSORBOARD_LOG + "/outputs/eval_logs/",
        eval_freq=100,
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
        policy_kwargs=UIC_POLICY_KWARGS,
        vf_coef=UIC_VF_COEF,
        target_kl=UIC_TARGET_KL,
        tensorboard_log=UIC_TENSORBOARD_LOG,
    )

    agent.train(
        total_timesteps=TOTAL_TIME_STEPS,
        callback=eval_callback,
    )

    train_envs.save(
        UIC_TENSORBOARD_LOG + "/outputs/" + community_id + "_env_train_normalize.pkl"
    )

    agent.model.save(
        UIC_TENSORBOARD_LOG + "/outputs/" + community_id + "_user_incentive_coordinator"
    )

    print("Training completed.")


def train_all_uics():
    for community_id in COMMUNITY_IDS:
        print(f"Training UIC for community {community_id}...")
        train_uic(community_id)
        print(f"UIC training completed for community {community_id}.\n")


if __name__ == "__main__":
    train_all_uics()
    # train_uic(COMMUNITY_IDS[0])
