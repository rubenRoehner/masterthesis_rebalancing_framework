"""
hyperparameter_optimization_uic.py

Hyperparameter optimization for User Incentive Coordinator (UIC).
This script uses Optuna to perform Bayesian optimization of hyperparameters
for the PPO-based User Incentive Coordinator agent in e-scooter rebalancing.
"""

from datetime import datetime, timedelta
import torch
import pandas as pd
import gymnasium as gym
import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial

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

from uic_training_loop import USER_WILLINGNESS_FN

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import csv
import os

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

study_filename = f"uic_ho_{timestamp}"
output_dir = "ho_results"
os.makedirs(output_dir, exist_ok=True)
N_TRIALS = 30

# global parameters
COMMUNITY_ID = "861faa7afffffff"
FLEET_SIZE = 40
N_EPOCHS = 20
MAX_STEPS_PER_EPISODE = 256
TOTAL_TIME_STEPS = 30_000
START_TIME = datetime(2025, 2, 11, 14, 0)

N_WORKERS = 1
BASE_SEED = 42

# UIC parameters
UIC_STEP_DURATION = 60  # in minutes

REWARD_WEIGHT_DEMAND = 1.0
REWARD_WEIGHT_REBALANCING = 0.5
REWARD_WEIGHT_GINI = 0.25

UIC_POLICY = "MultiInputPolicy"
UIC_N_STEPS = 64
UIC_LEARNING_RATE = 3e-4
UIC_GAMMA = 0.918
UIC_GAE_LAMBDA = 0.95
UIC_CLIP_RANGE = 0.29
UIC_ENT_COEF = 0.07
UIC_BATCH_SIZE = 64
UIC_VERBOSE = 1
UIC_TENSORBOARD_LOG = "rl_framework/runs/"

ZONE_COMMUNITY_MAP: pd.DataFrame = pd.read_pickle(
    "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
)

NUM_COMMUNITIES = ZONE_COMMUNITY_MAP["community_index"].nunique()
N_TOTAL_ZONES = ZONE_COMMUNITY_MAP.shape[0]

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
    neighbors = [
        neighbor for neighbor in row["neighbors"] if neighbor in COMMUNTIY_ZONE_IDS
    ]
    ZONE_NEIGHBOR_MAP.update({row["grid_index"]: neighbors})

N_ZONES = ZONE_COMMUNITY_MAP[
    ZONE_COMMUNITY_MAP["community_index"] == COMMUNITY_ID
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

torch.cuda.set_device(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_trial_callback(study: Study, trial: FrozenTrial) -> None:
    """Save trial results to CSV file after each trial completion.

    Args:
        study: Optuna study object
        trial: completed trial object with parameters and results

    Returns:
        None

    Raises:
        None
    """
    csv_path = os.path.join(output_dir, f"{study_filename}.csv")
    header = ["trial_number"] + list(trial.params.keys()) + ["value"]
    row = [trial.number] + list(trial.params.values()) + [trial.value]
    write_header = trial.number == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


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
        seed: base random seed for environment

    Returns:
        callable: environment factory function

    Raises:
        None
    """

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


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization.

    Defines the search space for UIC hyperparameters and evaluates
    agent performance using the suggested parameter combination.

    Args:
        trial: Optuna trial object for suggesting hyperparameters

    Returns:
        float: best mean reward achieved during evaluation

    Raises:
        None
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 2e-4, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 512])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    gamma = trial.suggest_float("gamma", 0.88, 0.95, step=0.001)
    clip_range = trial.suggest_float("clip_range", 0.12, 0.22, step=0.01)
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0, step=0.1)

    raw_target_kl = trial.suggest_categorical(
        "use_target_kl", [None, 0.005, 0.01, 0.02]
    )
    if raw_target_kl is None:
        target_kl: float | None = None
    else:
        target_kl = float(raw_target_kl)

    n_layers = trial.suggest_int("n_layers", 2, 3)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    activation_name = trial.suggest_categorical(
        "activation", ["ReLU", "Tanh", "LeakyReLU"]
    )
    if activation_name == "ReLU":
        activation_fn = torch.nn.ReLU
    elif activation_name == "Tanh":
        activation_fn = torch.nn.Tanh
    else:
        activation_fn = torch.nn.LeakyReLU

    policy_kwargs = {
        "net_arch": dict(
            pi=[hidden_size] * n_layers,
            vf=[hidden_size] * n_layers,
        ),
        "activation_fn": activation_fn,
    }

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
        eval_freq=1000,
    )

    agent = UserIncentiveCoordinator(
        policy=UIC_POLICY,
        env=train_envs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        n_epochs=N_EPOCHS,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=UIC_GAE_LAMBDA,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=0,
        tensorboard_log=None,
        vf_coef=vf_coef,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
    )

    agent.train(
        total_timesteps=TOTAL_TIME_STEPS,
        callback=eval_callback,
    )

    return eval_callback.best_mean_reward


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        study_name="escooter_uic_hyperparameter_optimization",
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
        callbacks=[save_trial_callback],
    )

    df = study.trials_dataframe(
        attrs=("number", "params", "value", "datetime_start", "duration", "user_attrs")
    )
    results_path = os.path.join(output_dir, f"{study_filename}_final.pickle")
    df.to_pickle(results_path)

    print("Best hyperparameters:", study.best_params)
    print("Best mean reward:", study.best_value)
