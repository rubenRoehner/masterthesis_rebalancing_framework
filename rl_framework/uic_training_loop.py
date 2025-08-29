"""
uic_training_loop.py

This script orchestrates the parallel training of User Incentive Coordinator (UIC) agents
for multiple communities. It sets up the training environment, loads necessary data and
forecasters, and manages the distribution of training tasks across available GPUs.

The script can optionally load a pre-trained Regional Distribution Coordinator (RDC) agent
to simulate its influence on the global vehicle state during UIC training, providing a more
realistic and stable training environment.
"""

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

RDC_AGENT_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/runs/outputs/rdc_agent_model_20250801-185856.pth"

# --- Parallelization settings ---
AVAILABLE_GPUS = [1, 2, 3]
N_GPUS = len(AVAILABLE_GPUS)
BASE_SEED = 42

# --- UIC parameters ---
UIC_POLICY = "MultiInputPolicy"
UIC_STEP_DURATION = 60
REWARD_WEIGHT_DEMAND = 7.0
REWARD_WEIGHT_REBALANCING = 0.2
REWARD_WEIGHT_GINI = 7.0

# Optimized hyperparameters
UIC_N_STEPS = 512
UIC_LEARNING_RATE = 2.124e-05
UIC_GAMMA = 0.9629645139686621
UIC_GAE_LAMBDA = 0.95
UIC_CLIP_RANGE = 0.26
UIC_ENT_COEF = 0.00016
UIC_BATCH_SIZE = 32
UIC_VERBOSE = 0
UIC_POLICY_KWARGS = {
    "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
    "activation_fn": torch.nn.LeakyReLU,
}
UIC_VF_COEF = 0.68
UIC_TARGET_KL = 0.02
UIC_TENSORBOARD_LOG = "rl_framework/runs/UIC/"

# --- Data Paths ---
DROP_OFF_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_dropoff_demand_h3_hourly.pickle"
PICK_UP_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_pickup_demand_h3_hourly.pickle"
DROP_OFF_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_dropoff_forecasts.pkl"
PICK_UP_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_pickup_forecasts.pkl"
ZONE_COMMUNITY_MAP: pd.DataFrame = pd.read_pickle(
    "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
)
NUM_COMMUNITIES = ZONE_COMMUNITY_MAP["community_index"].nunique()
N_TOTAL_ZONES = ZONE_COMMUNITY_MAP.shape[0]


ZONE_INDEX_MAP: dict[str, int] = {}
for i, row in ZONE_COMMUNITY_MAP.iterrows():
    ZONE_INDEX_MAP.update({row["grid_index"]: i})

ZONE_NEIGHBOR_MAP_DF: pd.DataFrame = pd.read_pickle(
    "/home/ruroit00/rebalancing_framework/processed_data/grid_cells_neighbors_list.pickle"
)
ZONE_NEIGHBOR_MAP: dict[str, list[str]] = {}
for i, row in ZONE_NEIGHBOR_MAP_DF.iterrows():
    neighbors = [neighbor for neighbor in row["neighbors"]]
    ZONE_NEIGHBOR_MAP.update({row["grid_index"]: neighbors})

COMMUNITY_INDEX_MAP: dict[str, int] = {}
for index, value in enumerate(sorted(ZONE_COMMUNITY_MAP["community_index"].unique())):
    COMMUNITY_INDEX_MAP.update({value: index})


def USER_WILLINGNESS_FN(incentive: float) -> float:
    """
    User willingness function mapping incentive level to compliance probability.

    Args:
        incentive: The incentive value offered to the user.

    Returns:
        float: The probability of user compliance.

    Raises:
        None
    """
    return 0.4 * incentive


def make_env(
    rank: int,
    community_id: str,
    forecasters: dict,
    providers: dict,
    rdc_agent_instance: RegionalDistributionCoordinator,
    device: torch.device,
    seed: int = 0,
):
    """
    Factory for creating the unified training environment.

    This function returns a callable that, when called, creates and initializes
    an instance of the EscooterUICEnv, wraps it with a Monitor, and sets a unique
    seed for reproducibility.

    Args:
        rank: The rank of the process, used for seeding.
        community_id: The ID of the community for this environment instance.
        forecasters: A dictionary of demand forecasters.
        providers: A dictionary of demand providers.
        rdc_agent_instance: An instance of the RDC agent.
        device: The PyTorch device to use.
        seed: The base seed for the environment.

    Returns:
        A callable function that creates and returns a configured gym environment.

    Raises:
        None
    """

    def _init():
        """
        Initializes and returns a monitored gym environment.

        Args:
            None

        Returns:
            gym.Env: The initialized and monitored environment.

        Raises:
            None
        """
        env = EscooterUICEnv(
            community_id=community_id,
            fleet_size=GLOBAL_FLEET_SIZE,
            zone_community_map=ZONE_COMMUNITY_MAP,
            community_index_map=COMMUNITY_INDEX_MAP,
            zone_index_map=ZONE_INDEX_MAP,
            zone_neighbor_map=ZONE_NEIGHBOR_MAP,
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
    """
    Trains a single UIC agent for a specific community on a specific GPU.

    This function sets up the environment, initializes the agent, and runs the
    training loop. After training, it saves the trained model and the
    VecNormalize statistics.

    Args:
        args: A tuple containing (community_id, gpu_id, rdc_agent_instance).

    Returns:
        str: The community_id of the agent that was trained.

    Raises:
        None
    """
    community_id, gpu_id, rdc_agent_instance = args
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Starting training for community {community_id} on device {device}")

    if rdc_agent_instance:
        rdc_agent_instance.to(device)

    forecasters = {
        "pickup": IrConvLstmDemandPreForecaster(
            num_communities=len(COMMUNITY_IDS),
            num_zones=N_TOTAL_ZONES,
            zone_community_map=ZONE_COMMUNITY_MAP,
            demand_data_path=PICK_UP_DEMAND_FORECAST_DATA_PATH,
        ),
        "dropoff": IrConvLstmDemandPreForecaster(
            num_communities=len(COMMUNITY_IDS),
            num_zones=N_TOTAL_ZONES,
            zone_community_map=ZONE_COMMUNITY_MAP,
            demand_data_path=DROP_OFF_DEMAND_FORECAST_DATA_PATH,
        ),
    }
    providers = {
        "pickup": DemandProviderImpl(
            num_communities=len(COMMUNITY_IDS),
            num_zones=N_TOTAL_ZONES,
            zone_community_map=ZONE_COMMUNITY_MAP,
            demand_data_path=PICK_UP_DEMAND_DATA_PATH,
            startTime=START_TIME,
            endTime=END_TIME,
        ),
        "dropoff": DemandProviderImpl(
            num_communities=len(COMMUNITY_IDS),
            num_zones=N_TOTAL_ZONES,
            zone_community_map=ZONE_COMMUNITY_MAP,
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
    """
    Main parallel training function.

    This function loads a pre-trained RDC agent (if specified), prepares the
    training tasks for each community, and distributes them across multiple GPUs
    using a multiprocessing pool.

    Args:
        None

    Returns:
        None

    Raises:
        RuntimeError: If the multiprocessing start method cannot be set to 'spawn'.
                      The function catches this and proceeds with the default method.
    """

    rdc_agent_instance = None
    if RDC_AGENT_PATH:
        print(f"Loading RDC agent from {RDC_AGENT_PATH}...")
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
        (community_id, AVAILABLE_GPUS[i % N_GPUS], rdc_agent_instance)
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
