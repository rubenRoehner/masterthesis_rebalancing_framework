"""
hyperparameter_optimization_rdc.py

Hyperparameter optimization for Regional Distribution Coordinator (RDC).
This script uses Optuna to perform Bayesian optimization of hyperparameters
for the multi-head DQN-based Regional Distribution Coordinator agent in e-scooter fleet management.
"""

import sys
import os

# Add the parent directories to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import torch
import pandas as pd
import numpy as np
import optuna
from optuna.study import Study
from optuna.trial import FrozenTrial
import csv
import os

from regional_distribution_coordinator.regional_distribution_coordinator import (
    RegionalDistributionCoordinator,
)
from regional_distribution_coordinator.escooter_rdc_env import EscooterRDCEnv
from demand_forecasting.IrConv_LSTM_pre_forecaster import (
    IrConvLstmDemandPreForecaster,
)
from demand_provider.demand_provider_impl import DemandProviderImpl


OPTIMIZE_LEARNING_RATE = True
OPTIMIZE_ARCHITECTURE = True
OPTIMIZE_REPLAY_BUFFER = True
OPTIMIZE_EXPLORATION = True

FLAG_LABELS = {
    "OPTIMIZE_REPLAY_BUFFER": "replaybuffer",
    "OPTIMIZE_ARCHITECTURE": "architecture",
    "OPTIMIZE_LEARNING_RATE": "learningrate",
    "OPTIMIZE_EXPLORATION": "exploration",
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

study_filename = f"rdc_ho_{study_label}_{timestamp}"

N_TRIALS = 100

# global parameters
FLEET_SIZE = 810
N_TRAINING_EPISODES = 100
N_EVAL_EPISODES = 20
MAX_STEPS_PER_EPISODE = 100
START_TIME = datetime(2025, 2, 11, 14, 0)
END_TIME = datetime(2025, 5, 4, 15, 0)
EVAL_END_TIME = datetime(2025, 5, 18, 15, 0)

output_dir = "ho_results"
os.makedirs(output_dir, exist_ok=True)

ZONE_COMMUNITY_MAP: pd.DataFrame = pd.read_pickle(
    "/home/ruroit00/rebalancing_framework/processed_data/grid_community_map.pickle"
)

N_COMMUNITIES = ZONE_COMMUNITY_MAP["community_index"].nunique()
N_ZONES = ZONE_COMMUNITY_MAP.shape[0]

COMMUNITY_INDEX_MAP: dict[str, int] = {}
for index, value in enumerate(sorted(ZONE_COMMUNITY_MAP["community_index"].unique())):
    COMMUNITY_INDEX_MAP.update({value: index})
print(f"Community Index Map: {COMMUNITY_INDEX_MAP}")

ZONE_INDEX_MAP: dict[str, int] = {}
for i, row in ZONE_COMMUNITY_MAP.iterrows():
    ZONE_INDEX_MAP.update({row["grid_index"]: i})

RDC_ACTION_VALUES = [-8, -4, -2, 0, 2, 4, 8]
RDC_FEATURES_PER_COMMUNITY = 3

# --- REWARD WEIGHTS ---
RDC_REWARD_WEIGHT_DEMAND = 7.0
RDC_REWARD_WEIGHT_REBALANCING = 0.2
RDC_REWARD_WEIGHT_GINI = 7.0


DROP_OFF_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_dropoff_demand_h3_hourly.pickle"
PICK_UP_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_pickup_demand_h3_hourly.pickle"

DROP_OFF_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_dropoff_forecasts.pkl"
PICK_UP_DEMAND_FORECAST_DATA_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/data/IrConv_LSTM_pickup_forecasts.pkl"

# --- INITIALIZE ENVIRONMENT ---
dropoff_demand_forecaster = IrConvLstmDemandPreForecaster(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=DROP_OFF_DEMAND_FORECAST_DATA_PATH,
)

pickup_demand_forecaster = IrConvLstmDemandPreForecaster(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=PICK_UP_DEMAND_FORECAST_DATA_PATH,
)

dropoff_demand_provider = DemandProviderImpl(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
    startTime=START_TIME,
    endTime=END_TIME,
)

pickup_demand_provider = DemandProviderImpl(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=PICK_UP_DEMAND_DATA_PATH,
    startTime=START_TIME,
    endTime=END_TIME,
)

eval_dropoff_demand_provider = DemandProviderImpl(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
    startTime=END_TIME,
    endTime=EVAL_END_TIME,
)

eval_pickup_demand_provider = DemandProviderImpl(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=PICK_UP_DEMAND_DATA_PATH,
    startTime=END_TIME,
    endTime=EVAL_END_TIME,
)

torch.cuda.set_device(3)
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


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization.

    Defines the search space for RDC hyperparameters based on optimization flags
    and evaluates agent performance using the suggested parameter combination.
    The search space includes replay buffer parameters, network architecture,
    learning rates, exploration parameters, and reward weights.

    Args:
        trial: Optuna trial object for suggesting hyperparameters

    Returns:
        float: mean reward achieved during evaluation episodes

    Raises:
        None
    """
    # --- HYPERPARAMETERS ---
    if OPTIMIZE_REPLAY_BUFFER:
        RDC_REPLAY_BUFFER_CAPACITY = trial.suggest_int(
            "rdc_replay_buffer_capacity", 5_000, 70_000, log=True
        )
        RDC_REPLAY_BUFFER_ALPHA = trial.suggest_float(
            "rdc_replay_buffer_alpha", 0.2, 1.0, step=0.1
        )
        RDC_REPLAY_BUFFER_BETA_START = trial.suggest_float(
            "rdc_replay_buffer_beta_start", 0.25, 0.56, step=0.01
        )
        RDC_REPLAY_BUFFER_BETA_FRAMES = trial.suggest_int(
            "rdc_replay_buffer_beta_frames", 5_000, 65_000, log=True
        )
        RDC_TAU = trial.suggest_float("rdc_tau", 1e-4, 1e-2, log=True)
    else:
        RDC_REPLAY_BUFFER_CAPACITY = 33_000
        RDC_REPLAY_BUFFER_ALPHA = 1.0
        RDC_REPLAY_BUFFER_BETA_START = 0.43
        RDC_REPLAY_BUFFER_BETA_FRAMES = 59_500
        RDC_TAU = 0.009327

    if OPTIMIZE_ARCHITECTURE:
        RDC_BATCH_SIZE = trial.suggest_categorical("rdc_batch_size", [256, 512, 1024])
        RDC_HIDDEN_DIM = trial.suggest_categorical("rdc_hidden_dim", [256, 512, 1024])
    else:
        RDC_BATCH_SIZE = 256
        RDC_HIDDEN_DIM = 512

    if OPTIMIZE_LEARNING_RATE:
        RDC_LR = trial.suggest_float("rdc_lr", 1e-6, 1e-5, log=True)
        RDC_LR_STEP_SIZE = trial.suggest_int("rdc_lr_step_size", 1_000, 6_000, step=100)
        RDC_LR_GAMMA = trial.suggest_float("rdc_lr_gamma", 0.7, 0.8, step=0.01)
        RDC_GAMMA = trial.suggest_float("rdc_gamma", 0.96, 0.99, step=0.001)
    else:
        RDC_LR = 2.27e-05
        RDC_LR_STEP_SIZE = 1250
        RDC_LR_GAMMA = 0.75
        RDC_GAMMA = 0.97

    RDC_EPSILON_START = 1.0
    if OPTIMIZE_EXPLORATION:
        RDC_EPSILON_END = trial.suggest_float(
            "rdc_epsilon_end", 0.06, 0.075, step=0.001
        )
        RDC_EPSILON_DECAY = trial.suggest_float(
            "rdc_epsilon_decay", 0.96, 0.975, log=True
        )
    else:
        RDC_EPSILON_END = 0.06
        RDC_EPSILON_DECAY = 0.961

    RDC_STEP_DURATION = 60

    train_envs = EscooterRDCEnv(
        num_communities=N_COMMUNITIES,
        n_zones=N_ZONES,
        action_values=RDC_ACTION_VALUES,
        max_steps=MAX_STEPS_PER_EPISODE,
        step_duration=timedelta(minutes=RDC_STEP_DURATION),
        start_time=START_TIME,
        reward_weight_demand=RDC_REWARD_WEIGHT_DEMAND,
        reward_weight_rebalancing=RDC_REWARD_WEIGHT_REBALANCING,
        reward_weight_gini=RDC_REWARD_WEIGHT_GINI,
        zone_community_map=ZONE_COMMUNITY_MAP,
        zone_index_map=ZONE_INDEX_MAP,
        community_index_map=COMMUNITY_INDEX_MAP,
        dropoff_demand_forecaster=dropoff_demand_forecaster,
        pickup_demand_forecaster=pickup_demand_forecaster,
        dropoff_demand_provider=dropoff_demand_provider,
        pickup_demand_provider=pickup_demand_provider,
        device=device,
        fleet_size=FLEET_SIZE,
    )
    train_envs.reset(seed=1)

    eval_env = EscooterRDCEnv(
        num_communities=N_COMMUNITIES,
        n_zones=N_ZONES,
        action_values=RDC_ACTION_VALUES,
        max_steps=MAX_STEPS_PER_EPISODE,
        step_duration=timedelta(minutes=RDC_STEP_DURATION),
        start_time=START_TIME,
        reward_weight_demand=RDC_REWARD_WEIGHT_DEMAND,
        reward_weight_rebalancing=RDC_REWARD_WEIGHT_REBALANCING,
        reward_weight_gini=RDC_REWARD_WEIGHT_GINI,
        zone_community_map=ZONE_COMMUNITY_MAP,
        zone_index_map=ZONE_INDEX_MAP,
        community_index_map=COMMUNITY_INDEX_MAP,
        dropoff_demand_forecaster=dropoff_demand_forecaster,
        pickup_demand_forecaster=pickup_demand_forecaster,
        dropoff_demand_provider=eval_dropoff_demand_provider,
        pickup_demand_provider=eval_pickup_demand_provider,
        device=device,
        fleet_size=FLEET_SIZE,
    )
    eval_env.reset(seed=2)

    rdc_agent = RegionalDistributionCoordinator(
        state_dim=N_COMMUNITIES * RDC_FEATURES_PER_COMMUNITY,
        num_communities=N_COMMUNITIES,
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
        tau=RDC_TAU,
        device=device,
    )

    # --- TRAINING LOOP ---
    for episode in range(N_TRAINING_EPISODES):
        current_observation, _ = train_envs.reset()
        for step in range(MAX_STEPS_PER_EPISODE):
            action = rdc_agent.select_action(current_observation)

            next_observation, reward, terminated, truncated, _ = train_envs.step(action)
            done = terminated or truncated

            rdc_agent.store_experience(
                current_observation, action, reward, next_observation, done
            )

            rdc_agent.train()
            current_observation = next_observation

            if done:
                break

    train_envs.close()

    # --- EVALUATION LOOP ---
    rdc_agent.epsilon = 0
    rewards = []
    for episode in range(N_EVAL_EPISODES):
        current_observation, _ = eval_env.reset()
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = rdc_agent.select_action(current_observation)
            next_observation, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

            total_reward += reward
            current_observation = next_observation

            if done:
                break

        rewards.append(total_reward)

    eval_env.close()

    return float(np.mean(rewards))


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        study_name="escooter_rdc_hyperparameter_optimization",
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=10,
        ),
    )

    study.set_user_attr("optimization_target", study_label)

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=5,
        callbacks=[save_trial_callback],
    )

    df = study.trials_dataframe(
        attrs=("number", "params", "value", "datetime_start", "duration", "user_attrs")
    )
    results_path = os.path.join(output_dir, f"{study_filename}_final.pickle")
    df.to_pickle(results_path)

    print("Best trial:")
    print(study.best_trial)
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
