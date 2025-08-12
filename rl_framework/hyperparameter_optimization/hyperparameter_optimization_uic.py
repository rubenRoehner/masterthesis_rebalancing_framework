"""
hyperparameter_optimization_uic.py

Hyperparameter optimization for User Incentive Coordinator (UIC).
This script uses Optuna to perform Bayesian optimization of hyperparameters
for the PPO-based User Incentive Coordinator agent in e-scooter rebalancing.
"""

import sys
import os

# Add the parent directories to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_framework.regional_distribution_coordinator.regional_distribution_coordinator import (
    RegionalDistributionCoordinator,
)
from training_loop import RDC_ACTION_VALUES, RDC_HIDDEN_DIM, RDC_FEATURES_PER_COMMUNITY

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

OPTIMIZE_PPO_CORE = True
OPTIMIZE_ARCHITECTURE = True
OPTIMIZE_STABILITY = True

FLAG_LABELS = {
    "OPTIMIZE_PPO_CORE": "ppo-core",
    "OPTIMIZE_ARCHITECTURE": "architecture",
    "OPTIMIZE_STABILITY": "stability",
}

active_flags = [
    label for flag, label in FLAG_LABELS.items() if globals().get(flag, False)
]
if not active_flags:
    study_label = "default"
elif len(active_flags) == len(FLAG_LABELS):
    study_label = "final_joint"
else:
    study_label = active_flags[0]

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

study_filename = f"uic_ho_{study_label}_{timestamp}"
output_dir = "ho_results"
os.makedirs(output_dir, exist_ok=True)
N_TRIALS = 100

# global parameters
COMMUNITY_ID = "861faa7afffffff"
GLOBAL_FLEET_SIZE = 810
N_EPOCHS = 15
MAX_STEPS_PER_EPISODE = 256
TOTAL_TIME_STEPS = 40_000
START_TIME = datetime(2025, 2, 11, 14, 0)
END_TIME = datetime(2025, 5, 4, 15, 0)
EVAL_END_TIME = datetime(2025, 5, 18, 15, 0)

N_WORKERS = 1
BASE_SEED = 42

# UIC parameters
UIC_STEP_DURATION = 60  # in minutes

REWARD_WEIGHT_DEMAND = 7.0
REWARD_WEIGHT_REBALANCING = 0.2
REWARD_WEIGHT_GINI = 7.0

UIC_POLICY = "MultiInputPolicy"

UIC_LEARNING_RATE = 3.9876e-05
UIC_N_STEPS = 256
UIC_BATCH_SIZE = 64
UIC_CLIP_RANGE = 0.3
UIC_ENT_COEF = 9.8e-05
UIC_VF_COEF = 0.7

UIC_GAMMA = 0.9638746
UIC_GAE_LAMBDA = 0.97
UIC_TARGET_KL = 0.02

UIC_POLICY_KWARGS = {
    "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
    "activation_fn": torch.nn.ReLU,
}

UIC_VERBOSE = 1
UIC_TENSORBOARD_LOG = "rl_framework/runs/"

RDC_AGENT_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/runs/outputs/rdc_agent_model_20250801-185856.pth"

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
    startTime=START_TIME,
    endTime=END_TIME,
)

pickup_demand_provider = DemandProviderImpl(
    num_communities=NUM_COMMUNITIES,
    num_zones=N_TOTAL_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=PICK_UP_DEMAND_DATA_PATH,
    startTime=START_TIME,
    endTime=END_TIME,
)

eval_dropoff_demand_provider = DemandProviderImpl(
    num_communities=NUM_COMMUNITIES,
    num_zones=N_TOTAL_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
    startTime=END_TIME,
    endTime=EVAL_END_TIME,
)
eval_pickup_demand_provider = DemandProviderImpl(
    num_communities=NUM_COMMUNITIES,
    num_zones=N_TOTAL_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=PICK_UP_DEMAND_DATA_PATH,
    startTime=END_TIME,
    endTime=EVAL_END_TIME,
)

torch.cuda.set_device(1)
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
        rdc_agent_instance = RegionalDistributionCoordinator(
            device=device,
            hidden_dim=RDC_HIDDEN_DIM,
            action_values=RDC_ACTION_VALUES,
            num_communities=NUM_COMMUNITIES,
            state_dim=NUM_COMMUNITIES * RDC_FEATURES_PER_COMMUNITY,
        )
        rdc_network = torch.load(RDC_AGENT_PATH, map_location=device)
        rdc_agent_instance.set_evaluation_mode(rdc_network)
        env: gym.Env = EscooterUICEnv(
            community_id=COMMUNITY_ID,
            fleet_size=GLOBAL_FLEET_SIZE,
            dropoff_demand_forecaster=dropoff_demand_forecaster,
            pickup_demand_forecaster=pickup_demand_forecaster,
            dropoff_demand_provider=dropoff_demand_provider,
            pickup_demand_provider=pickup_demand_provider,
            device=device,
            zone_neighbor_map=zone_neighbor_map,
            zone_index_map=zone_index_map,
            user_willingness_fn=USER_WILLINGNESS_FN,
            max_steps=MAX_STEPS_PER_EPISODE,
            start_time=START_TIME,
            step_duration=timedelta(minutes=UIC_STEP_DURATION),
            reward_weight_demand=REWARD_WEIGHT_DEMAND,
            reward_weight_rebalancing=REWARD_WEIGHT_REBALANCING,
            reward_weight_gini=REWARD_WEIGHT_GINI,
            rdc_agent=rdc_agent_instance,
            zone_community_map=ZONE_COMMUNITY_MAP,
            community_index_map=COMMUNITY_INDEX_MAP,
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
    if OPTIMIZE_PPO_CORE:
        learning_rate = trial.suggest_float("learning_rate", 2e-5, 6e-5, log=True)
        n_steps = trial.suggest_categorical("n_steps", [256, 512])
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        clip_range = trial.suggest_float("clip_range", 0.25, 0.3, step=0.01)
        ent_coef = trial.suggest_float("ent_coef", 2e-5, 2e-4, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.65, 0.75, step=0.01)
    else:
        learning_rate = UIC_LEARNING_RATE
        n_steps = UIC_N_STEPS
        batch_size = UIC_BATCH_SIZE
        clip_range = UIC_CLIP_RANGE
        ent_coef = UIC_ENT_COEF
        vf_coef = UIC_VF_COEF

    if OPTIMIZE_ARCHITECTURE:
        n_layers = 2
        hidden_size = trial.suggest_categorical("hidden_size", [128, 256])
        activation_name = trial.suggest_categorical(
            "activation", ["ReLU", "LeakyReLU"]
        )
        if activation_name == "ReLU":
            activation_fn = torch.nn.ReLU
        elif activation_name == "Tanh":
            activation_fn = torch.nn.Tanh
        elif activation_name == "SiLU":
            activation_fn = torch.nn.SiLU
        else:
            activation_fn = torch.nn.LeakyReLU

        policy_kwargs = {
            "net_arch": dict(
                pi=[hidden_size] * n_layers,
                vf=[hidden_size] * n_layers,
            ),
            "activation_fn": activation_fn,
        }
    else:
        policy_kwargs = UIC_POLICY_KWARGS

    if OPTIMIZE_STABILITY:
        gamma = trial.suggest_float("gamma", 0.95, 0.98, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.94, 0.97, step=0.01)

        raw_target_kl = trial.suggest_categorical(
            "use_target_kl", [0.02]
        )
        if raw_target_kl is None:
            target_kl: float | None = None
        else:
            target_kl = float(raw_target_kl)
    else:
        gamma = UIC_GAMMA
        gae_lambda = UIC_GAE_LAMBDA
        target_kl = None


    rdc_agent_instance = RegionalDistributionCoordinator(
        device=device,
        hidden_dim=RDC_HIDDEN_DIM,
        action_values=RDC_ACTION_VALUES,
        num_communities=NUM_COMMUNITIES,
        state_dim=NUM_COMMUNITIES * RDC_FEATURES_PER_COMMUNITY,
    )
    rdc_network = torch.load(RDC_AGENT_PATH, map_location=device)
    rdc_agent_instance.set_evaluation_mode(rdc_network)

    escooter_env = EscooterUICEnv(
        community_id=COMMUNITY_ID,
        fleet_size=GLOBAL_FLEET_SIZE,
        dropoff_demand_forecaster=dropoff_demand_forecaster,
        pickup_demand_forecaster=pickup_demand_forecaster,
        dropoff_demand_provider=eval_dropoff_demand_provider,
        pickup_demand_provider=eval_pickup_demand_provider,
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
        rdc_agent=rdc_agent_instance,
        zone_community_map=ZONE_COMMUNITY_MAP,
        community_index_map=COMMUNITY_INDEX_MAP,
    )

    train_envs = SubprocVecEnv(
        [
            make_env(
                rank=i,
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
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=0,
        tensorboard_log=None,
        vf_coef=vf_coef,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        device=device,
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
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5,
        ),
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=4,
        callbacks=[save_trial_callback],
        show_progress_bar=True,
    )

    df = study.trials_dataframe(
        attrs=("number", "params", "value", "datetime_start", "duration", "user_attrs")
    )
    results_path = os.path.join(output_dir, f"{study_filename}_final.pickle")
    df.to_pickle(results_path)

    print("Best hyperparameters:", study.best_params)
    print("Best mean reward:", study.best_value)
