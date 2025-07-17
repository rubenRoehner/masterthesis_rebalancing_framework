from datetime import datetime, timedelta
import torch
import pandas as pd
import multiprocessing as mp

from demand_forecasting.IrConv_LSTM_pre_forecaster import (
    IrConvLstmDemandPreForecaster,
)
from demand_provider.demand_provider_impl import DemandProviderImpl
from user_incentive_coordinator.escooter_uic_env import EscooterUICEnv
from user_incentive_coordinator.user_incentive_coordinator import (
    UserIncentiveCoordinator,
)
from regional_distribution_coordinator.regional_distribution_coordinator import (
    RegionalDistributionCoordinator,
)
from training_loop import RDC_ACTION_VALUES, RDC_HIDDEN_DIM, RDC_FEATURES_PER_COMMUNITY

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
GLOBAL_FLEET_SIZE = 810
N_EPOCHS = 20
MAX_STEPS_PER_EPISODE = 256
TOTAL_TIME_STEPS = 1_000_000
START_TIME = datetime(2025, 2, 11, 14, 0)
END_TIME = datetime(2025, 5, 18, 15, 0)

RDC_AGENT_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/runs/outputs/rdc_agent_model_20250713-220126.pth"

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

# --- Data Paths ---
DROP_OFF_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_dropoff_demand_h3_hourly.pickle"
PICK_UP_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_pickup_demand_h3_hourly.pickle"
DROP_OFF_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_dropoff_forecasts.pkl"
PICK_UP_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_pickup_forecasts.pkl"
ZONE_COMMUNITY_MAP_PATH = (
    "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
)
ZONE_NEIGHBOR_MAP_PATH = "/home/ruroit00/rebalancing_framework/processed_data/grid_cells_neighbors_list.pickle"


def USER_WILLINGNESS_FN(incentive: float) -> float:
    """User willingness function mapping incentive level to compliance probability."""
    return 0.4 * incentive


def make_env(
    rank: int,
    community_id: str,
    all_city_data: dict,
    forecasters: dict,
    providers: dict,
    rdc_agent_instance: RegionalDistributionCoordinator,
    device: torch.device,
    seed: int = 0,
):
    """Factory for creating the unified training environment."""

    def _init():
        env = EscooterUICEnv(
            community_id=community_id,
            fleet_size=all_city_data["fleet_size"],
            zone_community_map=all_city_data["zone_community_map"],
            community_index_map=all_city_data["community_index_map"],
            zone_index_map=all_city_data["zone_index_map"],
            zone_neighbor_map=all_city_data["zone_neighbor_map"],
            rdc_agent=rdc_agent_instance,
            user_willingness_fn=USER_WILLINGNESS_FN,
            pickup_demand_forecaster=forecasters["pickup"],
            dropoff_demand_forecaster=forecasters["dropoff"],
            pickup_demand_provider=providers["pickup"],
            dropoff_demand_provider=providers["dropoff"],
            max_steps=MAX_STEPS_PER_EPISODE,
            start_time=START_TIME,
            step_duration=timedelta(minutes=UIC_STEP_DURATION),
            reward_weight_demand=REWARD_WEIGHT_DEMAND,
            reward_weight_rebalancing=REWARD_WEIGHT_REBALANCING,
            reward_weight_gini=REWARD_WEIGHT_GINI,
            device=device,
        )
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env

    return _init


def train_uic(args):
    """Trains a single UIC agent for a specific community on a specific GPU."""
    community_id, gpu_id, all_city_data, rdc_agent_instance = args
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Starting training for community {community_id} on device {device}")

    if rdc_agent_instance:
        rdc_agent_instance.to(device)

    # Initialize forecasters and providers once per process
    forecasters = {
        "pickup": IrConvLstmDemandPreForecaster(
            num_communities=len(COMMUNITY_IDS),
            num_zones=all_city_data["n_total_zones"],
            zone_community_map=all_city_data["zone_community_map"],
            demand_data_path=PICK_UP_DEMAND_FORECAST_DATA_PATH,
        ),
        "dropoff": IrConvLstmDemandPreForecaster(
            num_communities=len(COMMUNITY_IDS),
            num_zones=all_city_data["n_total_zones"],
            zone_community_map=all_city_data["zone_community_map"],
            demand_data_path=DROP_OFF_DEMAND_FORECAST_DATA_PATH,
        ),
    }
    providers = {
        "pickup": DemandProviderImpl(
            num_communities=len(COMMUNITY_IDS),
            num_zones=all_city_data["n_total_zones"],
            zone_community_map=all_city_data["zone_community_map"],
            demand_data_path=PICK_UP_DEMAND_DATA_PATH,
            startTime=START_TIME,
            endTime=END_TIME,
        ),
        "dropoff": DemandProviderImpl(
            num_communities=len(COMMUNITY_IDS),
            num_zones=all_city_data["n_total_zones"],
            zone_community_map=all_city_data["zone_community_map"],
            demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
            startTime=START_TIME,
            endTime=END_TIME,
        ),
    }

    train_envs = DummyVecEnv(
        [
            make_env(
                rank=0,
                community_id=community_id,
                all_city_data=all_city_data,
                forecasters=forecasters,
                providers=providers,
                rdc_agent_instance=rdc_agent_instance,
                device=device,
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
        tensorboard_log=f"{UIC_TENSORBOARD_LOG}{community_id}",
    )

    agent.train(total_timesteps=TOTAL_TIME_STEPS, callback=None)

    model_save_path = f"{UIC_TENSORBOARD_LOG}outputs/{community_id}_uic_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    env_save_path = f"{UIC_TENSORBOARD_LOG}outputs/{community_id}_vecnormalize.pkl"

    agent.model.save(model_save_path)
    train_envs.save(env_save_path)

    print(
        f"--- Training completed for community {community_id}. Model saved to {model_save_path} ---"
    )
    return community_id


def train_all_uics_parallel() -> None:
    """Main parallel training function."""
    print("Loading all city-wide data into memory...")
    zone_community_map = pd.read_pickle(ZONE_COMMUNITY_MAP_PATH)
    zone_neighbor_map_df = pd.read_pickle(ZONE_NEIGHBOR_MAP_PATH)

    all_city_data = {
        "zone_community_map": zone_community_map,
        "zone_neighbor_map": {
            row["grid_index"]: row["neighbors"]
            for _, row in zone_neighbor_map_df.iterrows()
        },
        "zone_index_map": {
            row["grid_index"]: i for i, row in zone_community_map.iterrows()
        },
        "community_index_map": {cid: i for i, cid in enumerate(COMMUNITY_IDS)},
        "n_total_zones": len(zone_community_map),
        "fleet_size": GLOBAL_FLEET_SIZE,
    }

    rdc_agent_instance = None
    if RDC_AGENT_PATH:
        print(f"Loading RDC agent from {RDC_AGENT_PATH}...")
        # A dummy device is used for loading, it will be moved to the correct GPU in each process
        dummy_device = torch.device("cpu")
        rdc_agent_instance = RegionalDistributionCoordinator(
            device=dummy_device,
            hidden_dim=RDC_HIDDEN_DIM,
            action_values=RDC_ACTION_VALUES,
            num_communities=len(COMMUNITY_IDS),
            state_dim=len(COMMUNITY_IDS) * RDC_FEATURES_PER_COMMUNITY,
        )
        rdc_network = torch.load(RDC_AGENT_PATH, map_location=dummy_device)
        rdc_agent_instance.set_evaluation_mode(rdc_network)
    else:
        print("No RDC agent path provided. Training UICs without RDC simulation.")

    tasks = [
        (community_id, AVAILABLE_GPUS[i % N_GPUS], all_city_data, rdc_agent_instance)
        for i, community_id in enumerate(COMMUNITY_IDS)
    ]

    print(
        f"\nStarting parallel training for {len(COMMUNITY_IDS)} agents across {N_GPUS} GPUs..."
    )
    try:
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=N_GPUS) as pool:
            results = pool.map(train_uic, tasks)
    except RuntimeError:
        print("Could not set start method to 'spawn', proceeding with default.")
        with mp.Pool(processes=N_GPUS) as pool:
            results = pool.map(train_uic, tasks)

    print("\n--- ALL UIC TRAINING COMPLETED ---")
    for community_id in results:
        print(f"- Finished: {community_id}")


if __name__ == "__main__":
    train_all_uics_parallel()
