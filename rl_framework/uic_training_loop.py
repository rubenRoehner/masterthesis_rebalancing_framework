"""
uic_training_loop.py

Optimized training loop for User Incentive Coordinator (UIC) agents.
- Trains agents in parallel on specified GPUs.
- Loads all data only once to reduce I/O and memory overhead.
"""

from datetime import datetime, timedelta
import torch
import pandas as pd
import gymnasium as gym
import multiprocessing as mp

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

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# --- SCRIPT CONFIGURATION ---
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
FLEET_SIZE = 90
N_EPOCHS = 20
MAX_STEPS_PER_EPISODE = 256
TOTAL_TIME_STEPS = 200_000
START_TIME = datetime(2025, 2, 11, 14, 0)
END_TIME = datetime(2025, 5, 18, 15, 0)

# --- Parallelization settings ---
AVAILABLE_GPUS = [1, 2, 3]
N_GPUS = len(AVAILABLE_GPUS)
BASE_SEED = 42

# --- UIC parameters ---
UIC_POLICY = "MultiInputPolicy"
UIC_STEP_DURATION = 60
REWARD_WEIGHT_DEMAND = 1.0
REWARD_WEIGHT_REBALANCING = 5.0
REWARD_WEIGHT_GINI = 0.0

# Optimized hyperparameters
UIC_N_STEPS = 1024
UIC_LEARNING_RATE = 2.056573223956345e-06
UIC_GAMMA = 0.936
UIC_GAE_LAMBDA = 0.902
UIC_CLIP_RANGE = 0.2
UIC_ENT_COEF = 0.00011215783666166426
UIC_BATCH_SIZE = 32
UIC_VERBOSE = 0
UIC_POLICY_KWARGS = {
    "net_arch": dict(pi=[128, 128, 128], vf=[128, 128, 128]),
    "activation_fn": torch.nn.ReLU,
}
UIC_VF_COEF = 0.413
UIC_TARGET_KL = 0.022
UIC_TENSORBOARD_LOG = "rl_framework/runs/UIC/"


DROP_OFF_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_dropoff_demand_h3_hourly.pickle"
PICK_UP_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_pickup_demand_h3_hourly.pickle"

DROP_OFF_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_dropoff_forecasts.pkl"
PICK_UP_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_pickup_forecasts.pkl"


@staticmethod
def USER_WILLINGNESS_FN(incentive: float) -> float:
    """User willingness function mapping incentive level to compliance probability.

    Args:
        incentive: incentive level (0-1 normalized)

    Returns:
        float: probability that users will comply with the incentive

    Raises:
        None
    """
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
    """Create environment factory function for parallel training.

    Args:
        rank: environment rank for parallel execution
        n_zones: number of zones in the community
        dropoff_demand_forecaster: forecaster for dropoff demand patterns
        pickup_demand_forecaster: forecaster for pickup demand patterns
        dropoff_demand_provider: provider for actual dropoff demand data
        pickup_demand_provider: provider for actual pickup demand data
        device: PyTorch device for tensor operations
        zone_neighbor_map: mapping from zone IDs to neighbor zone IDs
        zone_index_map: mapping from zone IDs to indices
        community_id: ID of the community for this environment
        seed: base random seed for environment

    Returns:
        callable: environment factory function

    Raises:
        None
    """

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


def train_uic(args):
    """
    Train a UIC agent for a specific community on a specific GPU.
    """
    community_id, gpu_id, all_data = args
    zone_community_map = all_data["zone_community_map"]
    zone_neighbor_map_full = all_data["zone_neighbor_map"]

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Starting training for community {community_id} on device {device}")

    community_zones_df = zone_community_map[
        zone_community_map["community_index"] == community_id
    ]
    n_zones = len(community_zones_df)
    community_zone_ids = set(community_zones_df["grid_index"])

    zone_index_map = {row["grid_index"]: i for i, row in community_zones_df.iterrows()}

    zone_neighbor_map = {
        zone_id: [
            n
            for n in zone_neighbor_map_full.get(zone_id, [])
            if n in community_zone_ids
        ]
        for zone_id in community_zone_ids
    }

    dropoff_demand_forecaster = IrConvLstmDemandPreForecaster(
        num_communities=len(COMMUNITY_IDS),
        num_zones=all_data["n_total_zones"],
        zone_community_map=zone_community_map,
        demand_data_path=DROP_OFF_DEMAND_FORECAST_DATA_PATH,
    )
    pickup_demand_forecaster = IrConvLstmDemandPreForecaster(
        num_communities=len(COMMUNITY_IDS),
        num_zones=all_data["n_total_zones"],
        zone_community_map=zone_community_map,
        demand_data_path=PICK_UP_DEMAND_FORECAST_DATA_PATH,
    )
    dropoff_demand_provider = DemandProviderImpl(
        num_communities=len(COMMUNITY_IDS),
        num_zones=all_data["n_total_zones"],
        zone_community_map=zone_community_map,
        demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
        startTime=START_TIME,
        endTime=END_TIME,
    )
    pickup_demand_provider = DemandProviderImpl(
        num_communities=len(COMMUNITY_IDS),
        num_zones=all_data["n_total_zones"],
        zone_community_map=zone_community_map,
        demand_data_path=PICK_UP_DEMAND_DATA_PATH,
        startTime=START_TIME,
        endTime=END_TIME,
    )

    train_envs = DummyVecEnv(
        [
            make_env(
                rank=0,
                n_zones=n_zones,
                dropoff_demand_forecaster=dropoff_demand_forecaster,
                pickup_demand_forecaster=pickup_demand_forecaster,
                dropoff_demand_provider=dropoff_demand_provider,
                pickup_demand_provider=pickup_demand_provider,
                device=device,
                zone_neighbor_map=zone_neighbor_map,
                zone_index_map=zone_index_map,
                community_id=community_id,
                seed=BASE_SEED,
            )
        ]
    )

    train_envs = VecNormalize(
        train_envs, norm_obs=True, norm_reward=False, clip_obs=10.0
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
        device=device,
        tensorboard_log=UIC_TENSORBOARD_LOG,
    )

    agent.model.tensorboard_log = f"{UIC_TENSORBOARD_LOG}/{community_id}"

    agent.train(total_timesteps=TOTAL_TIME_STEPS, callback=None)

    model_save_path = f"{UIC_TENSORBOARD_LOG}/outputs/{community_id}_uic_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    env_save_path = f"{UIC_TENSORBOARD_LOG}/outputs/{community_id}_vecnormalize.pkl"

    agent.model.save(model_save_path)
    train_envs.save(env_save_path)

    print(
        f"--- Training completed for community {community_id}. Model saved to {model_save_path} ---"
    )
    return community_id


def train_all_uics_parallel() -> None:
    """
    Trains all UIC agents in parallel, assigning each to a different available GPU.
    """
    print("Loading all data into memory...")
    zone_community_map = pd.read_pickle(
        "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
    )
    zone_neighbor_map_df = pd.read_pickle(
        "/home/ruroit00/rebalancing_framework/processed_data/grid_cells_neighbors_list.pickle"
    )

    zone_neighbor_map_full = {
        row["grid_index"]: row["neighbors"]
        for _, row in zone_neighbor_map_df.iterrows()
    }

    all_data = {
        "zone_community_map": zone_community_map,
        "zone_neighbor_map": zone_neighbor_map_full,
        "n_total_zones": len(zone_community_map),
    }

    tasks = [
        (community_id, AVAILABLE_GPUS[i % N_GPUS], all_data)
        for i, community_id in enumerate(COMMUNITY_IDS)
    ]

    print(
        f"\nStarting parallel training for {len(COMMUNITY_IDS)} agents across {N_GPUS} GPUs: {AVAILABLE_GPUS}..."
    )
    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=N_GPUS) as pool:
        results = pool.map(train_uic, tasks)

    print("\n--- ALL UIC TRAINING COMPLETED ---")
    for community_id in results:
        print(f"- Finished: {community_id}")


if __name__ == "__main__":
    train_all_uics_parallel()
