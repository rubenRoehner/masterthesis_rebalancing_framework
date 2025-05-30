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
from demand_forecasting.IrConv_LSTM_demand_forecaster import (
    IrConvLstmDemandForecaster,
)
from demand_provider.demand_provider_impl import DemandProviderImpl


OPTIMIZE_REPLAY_BUFFER = False
OPTIMIZE_ARCHITECTURE = False
OPTIMIZE_LEARNING_RATE = False
OPTIMIZE_EXPLORATION = True
OPTIMIZE_REWARD_WEIGHTS = False

FLAG_LABELS = {
    "OPTIMIZE_REPLAY_BUFFER": "replaybuffer",
    "OPTIMIZE_ARCHITECTURE": "architecture",
    "OPTIMIZE_LEARNING_RATE": "learningrate",
    "OPTIMIZE_EXPLORATION": "exploration",
    "OPTIMIZE_REWARD_WEIGHTS": "rewardweights",
}

active_flags = [
    label for flag, label in FLAG_LABELS.items() if globals().get(flag, False)
]
if not active_flags:
    study_label = "default"
else:
    study_label = active_flags[0]

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

study_filename = f"rdc_ho_{study_label}_{timestamp}"

N_TRIALS = 20

# global parameters
FLEET_SIZE = 1000
N_TRAINING_EPISODES = 100
N_EVAL_EPISODES = 20
MAX_STEPS_PER_EPISODE = 100
START_TIME = datetime(2025, 2, 11, 14, 0)

output_dir = "ho_results"
os.makedirs(output_dir, exist_ok=True)

# [grid_id, community_id]
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

RDC_ACTION_VALUES = [-15, -10, -5, 0, 5, 10, 15]
RDC_FEATURES_PER_COMMUNITY = 3


DROP_OFF_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_dropoff_demand_h3_hourly.pickle"
PICK_UP_DEMAND_DATA_PATH = "/home/ruroit00/rebalancing_framework/processed_data/voi_pickup_demand_h3_hourly.pickle"

# --- INITIALIZE ENVIRONMENT ---
dropoff_demand_forecaster = IrConvLstmDemandForecaster(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    model_path="/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/models/irregular_convolution_LSTM_dropoff.pkl",
    demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
)

pickup_demand_forecaster = IrConvLstmDemandForecaster(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    model_path="/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/models/irregular_convolution_LSTM_pickup.pkl",
    demand_data_path=PICK_UP_DEMAND_DATA_PATH,
)

dropoff_demand_provider = DemandProviderImpl(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=DROP_OFF_DEMAND_DATA_PATH,
)

pickup_demand_provider = DemandProviderImpl(
    num_communities=N_COMMUNITIES,
    num_zones=N_ZONES,
    zone_community_map=ZONE_COMMUNITY_MAP,
    demand_data_path=PICK_UP_DEMAND_DATA_PATH,
)


def save_trial_callback(study: Study, trial: FrozenTrial):
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
    step_duration: int,
    rebalancing_cost: float,
    reward_weight_demand: float,
    reward_weight_rebalancing: float,
    reward_weight_gini: float,
    device: torch.device,
    seed: int,
):
    env = EscooterRDCEnv(
        num_communities=N_COMMUNITIES,
        n_zones=N_ZONES,
        action_values=RDC_ACTION_VALUES,
        max_steps=MAX_STEPS_PER_EPISODE,
        step_duration=timedelta(minutes=step_duration),
        start_time=START_TIME,
        operator_rebalancing_cost=rebalancing_cost,
        reward_weight_demand=reward_weight_demand,
        reward_weight_rebalancing=reward_weight_rebalancing,
        reward_weight_gini=reward_weight_gini,
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
    env.reset(seed=seed)
    return lambda: env


def objective(trial: optuna.Trial):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(trial.number % 4)

    # --- HYPERPARAMETERS ---
    if OPTIMIZE_REPLAY_BUFFER:
        RDC_REPLAY_BUFFER_CAPACITY = trial.suggest_int(
            "rdc_replay_buffer_capacity", 5_000, 20_000, step=5000
        )
        RDC_REPLAY_BUFFER_ALPHA = trial.suggest_float(
            "rdc_replay_buffer_alpha", 0.3, 0.8, step=0.1
        )
        RDC_REPLAY_BUFFER_BETA_START = trial.suggest_float(
            "rdc_replay_buffer_beta_start", 0.2, 0.6, step=0.1
        )
        RDC_REPLAY_BUFFER_BETA_FRAMES = trial.suggest_int(
            "rdc_replay_buffer_beta_frames", 50_000, 200_000, step=50_000
        )
    else:
        RDC_REPLAY_BUFFER_CAPACITY = 10_000
        RDC_REPLAY_BUFFER_ALPHA = 0.8
        RDC_REPLAY_BUFFER_BETA_START = 0.3
        RDC_REPLAY_BUFFER_BETA_FRAMES = 150_000

    if OPTIMIZE_ARCHITECTURE:
        RDC_BATCH_SIZE = trial.suggest_categorical(
            "rdc_batch_size", [64, 128, 256, 512]
        )
        RDC_HIDDEN_DIM = trial.suggest_categorical("rdc_hidden_dim", [64, 128, 256])
    else:
        RDC_BATCH_SIZE = 256
        RDC_HIDDEN_DIM = 256

    if OPTIMIZE_LEARNING_RATE:
        RDC_LR = trial.suggest_float("rdc_lr", 1e-6, 1e-4, log=True)
        RDC_LR_STEP_SIZE = trial.suggest_int("rdc_lr_step_size", 500, 2000, step=500)
        RDC_LR_GAMMA = trial.suggest_float("rdc_lr_gamma", 0.1, 0.9, step=0.1)
        RDC_GAMMA = trial.suggest_float("rdc_gamma", 0.9, 0.999)
    else:
        RDC_LR = 2.67e-6
        RDC_LR_STEP_SIZE = 1500
        RDC_LR_GAMMA = 0.9
        RDC_GAMMA = 0.926

    RDC_EPSILON_START = 1.0
    if OPTIMIZE_EXPLORATION:
        RDC_EPSILON_END = trial.suggest_float("rdc_epsilon_end", 0.01, 0.1, step=0.01)
        RDC_EPSILON_DECAY = trial.suggest_float(
            "rdc_epsilon_decay", 0.995, 0.9995, step=0.0005
        )
    else:
        RDC_EPSILON_END = 0.05
        RDC_EPSILON_DECAY = 0.999

    RDC_TAU = 0.001
    # RDC_TAU = trial.suggest_float("rdc_tau", 0.001, 0.01, step=0.001)

    RDC_STEP_DURATION = 60  # in minutes

    RDC_OPERATOR_REBALANCING_COST = 0.5

    if OPTIMIZE_REWARD_WEIGHTS:
        RDC_REWARD_WEIGHT_DEMAND = trial.suggest_float(
            "rdc_reward_weight_demand", 0.5, 1.5, step=0.1
        )
        RDC_REWARD_WEIGHT_REBALANCING = trial.suggest_float(
            "rdc_reward_weight_rebalancing", 0.5, 1.5, step=0.1
        )
        RDC_REWARD_WEIGHT_GINI = trial.suggest_float(
            "rdc_reward_weight_gini", 0.1, 0.5, step=0.05
        )
    else:
        RDC_REWARD_WEIGHT_DEMAND = 1.0
        RDC_REWARD_WEIGHT_REBALANCING = 1.0
        RDC_REWARD_WEIGHT_GINI = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_envs = EscooterRDCEnv(
        num_communities=N_COMMUNITIES,
        n_zones=N_ZONES,
        action_values=RDC_ACTION_VALUES,
        max_steps=MAX_STEPS_PER_EPISODE,
        step_duration=timedelta(minutes=RDC_STEP_DURATION),
        start_time=START_TIME,
        operator_rebalancing_cost=RDC_OPERATOR_REBALANCING_COST,
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
        operator_rebalancing_cost=RDC_OPERATOR_REBALANCING_COST,
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        study_name="escooter_rdc_hyperparameter_optimization",
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
