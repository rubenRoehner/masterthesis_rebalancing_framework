"""
evaluate_framework.py

Main evaluation script for the Hierarchical Reinforcement Learning framework.
This script loads trained RDC and UIC agents and evaluates their coordinated performance
in managing e-scooter fleet distribution across multiple communities.

Updated to support multiple seeded trials with results saved to pickle files.
"""

import sys
import os
import numpy as np
import random

# Add the parent directories to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demand_forecasting.IrConv_LSTM_pre_forecaster import (
    IrConvLstmDemandPreForecaster,
)
from regional_distribution_coordinator.regional_distribution_coordinator import (
    RegionalDistributionCoordinator,
)
from training_loop import RDC_ACTION_VALUES, RDC_HIDDEN_DIM, RDC_FEATURES_PER_COMMUNITY
from demand_provider.demand_provider_impl import DemandProviderImpl
from hrl_framework_evaluator import HRLFrameworkEvaluator
from stable_baselines3 import PPO
import torch
import pandas as pd
from datetime import timedelta, datetime
from collections import OrderedDict
from uic_training_loop import USER_WILLINGNESS_FN

# Evaluation parameters
NUM_TRIALS = 5
BASE_SEED = 42
OUTPUT_DIR = "./results"

# Mean in GBFS data is around 900 scooters
# [750, 900, 1200]
START_TIME = datetime(2025, 5, 18, 15, 0)
END_TIME = datetime(2025, 6, 18, 15, 0)
STEP_DURATION = 60
MAX_STEPS = 400

RDC_AGENT_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/runs/outputs/rdc_agent_model_20250801-185856.pth"

UIC_AGENT_PATHS = [
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa44fffffff_uic_model_20250811-152817.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa637ffffff_uic_model_20250811-151536.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa707ffffff_uic_model_20250811-155018.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa717ffffff_uic_model_20250811-200224.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa71fffffff_uic_model_20250811-200547.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa787ffffff_uic_model_20250811-204208.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa78fffffff_uic_model_20250812-003719.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa7a7ffffff_uic_model_20250812-004828.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa7afffffff_uic_model_20250812-012308.zip",
]

CONFIGURATIONS = [
    {
        "manual": True,
        "incentive": True,
        "name": "HRL_Both_Enabled_900",
        "fleet_size": 900,
    },
    {
        "manual": True,
        "incentive": False,
        "name": "HRL_Manual_Only_900",
        "fleet_size": 900,
    },
    {
        "manual": False,
        "incentive": True,
        "name": "HRL_Incentive_Only_900",
        "fleet_size": 900,
    },
    {
        "manual": True,
        "incentive": True,
        "name": "HRL_Both_Enabled_600",
        "fleet_size": 600,
    },
    {
        "manual": True,
        "incentive": False,
        "name": "HRL_Manual_Only_600",
        "fleet_size": 600,
    },
    {
        "manual": False,
        "incentive": True,
        "name": "HRL_Incentive_Only_600",
        "fleet_size": 600,
    },
    {
        "manual": True,
        "incentive": True,
        "name": "HRL_Both_Enabled_1200",
        "fleet_size": 1200,
    },
    {
        "manual": True,
        "incentive": False,
        "name": "HRL_Manual_Only_1200",
        "fleet_size": 1200,
    },
    {
        "manual": False,
        "incentive": True,
        "name": "HRL_Incentive_Only_1200",
        "fleet_size": 1200,
    },
]

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


torch.cuda.set_device(3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_trial(trial_idx: int, seed: int, rdc_agent, uic_agents, 
                    dropoff_demand_forecaster, pickup_demand_forecaster,
                    dropoff_demand_provider, pickup_demand_provider) -> pd.DataFrame:
    """Run a single trial of evaluation across all configurations.
    
    Args:
        trial_idx: Trial index number
        seed: Random seed for this trial
        rdc_agent: Trained RDC agent
        uic_agents: List of trained UIC agents
        dropoff_demand_forecaster: Demand forecaster for dropoffs
        pickup_demand_forecaster: Demand forecaster for pickups
        dropoff_demand_provider: Demand provider for dropoffs
        pickup_demand_provider: Demand provider for pickups
        
    Returns:
        DataFrame with results for all configurations in this trial
    """
    print(f"\n{'='*60}")
    print(f"Running Trial {trial_idx + 1} (Seed: {seed})")
    print(f"{'='*60}")
    
    set_seed(seed)
    
    trial_results = []
    
    for config in CONFIGURATIONS:
        print(f"\nEvaluating Configuration: {config['name']}")
        print(f"Manual Rebalancing: {config['manual']}")
        print(f"Incentive Rebalancing: {config['incentive']}")
        print(f"Fleet Size: {config['fleet_size']}")
        
        evaluator = HRLFrameworkEvaluator(
            fleet_size=config["fleet_size"],
            rdc_agent=rdc_agent,
            uic_agents=uic_agents,
            zone_community_map=ZONE_COMMUNITY_MAP,
            zone_neighbor_map=ZONE_NEIGHBOR_MAP,
            community_index_map=COMMUNITY_INDEX_MAP,
            dropoff_demand_forecaster=dropoff_demand_forecaster,
            pickup_demand_forecaster=pickup_demand_forecaster,
            dropoff_demand_provider=dropoff_demand_provider,
            pickup_demand_provider=pickup_demand_provider,
            start_time=START_TIME,
            max_steps=MAX_STEPS,
            step_duration=timedelta(minutes=STEP_DURATION),
            user_willingness_fn=USER_WILLINGNESS_FN,
            zone_index_map=ZONE_INDEX_MAP,
            device=device,
            enable_rdc_rebalancing=config["manual"],
            enable_uic_rebalancing=config["incentive"],
        )

        results = evaluator.evaluate()
        
        result_row = {
            'trial': trial_idx,
            'seed': seed,
            'configuration': config['name'],
            'manual_rebalancing': config['manual'],
            'incentive_rebalancing': config['incentive'],
            'fleet_size': config['fleet_size'],
            **results
        }
        
        trial_results.append(result_row)
        
        print(f"Results: Mean Satisfied Ratio = {results.get('mean_satisfied_ratio', 'N/A'):.4f}")
    
    return pd.DataFrame(trial_results)


def run_evaluation() -> None:
    """Run complete evaluation of the HRL framework across multiple trials.

    Loads trained RDC and UIC agents, sets up the evaluation environment,
    and measures the coordinated performance across multiple seeded trials.
    Uses constants defined at the top of the script for configuration.

    Returns:
        None
    """
    print("Starting HRL Framework Multi-Trial Evaluation")
    print(f"Number of trials: {NUM_TRIALS}")
    print(f"Base seed: {BASE_SEED}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rdc_agent_network: OrderedDict = torch.load(RDC_AGENT_PATH, map_location=device)
    uic_agents = [PPO.load(path, device=device) for path in UIC_AGENT_PATHS]

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

    rdc_agent = RegionalDistributionCoordinator(
        device=device,
        hidden_dim=RDC_HIDDEN_DIM,
        action_values=RDC_ACTION_VALUES,
        num_communities=NUM_COMMUNITIES,
        state_dim=NUM_COMMUNITIES * RDC_FEATURES_PER_COMMUNITY,
    )

    rdc_agent.set_evaluation_mode(rdc_agent_network)
    
    np.random.seed(BASE_SEED)
    trial_seeds = np.random.randint(0, 100000, size=NUM_TRIALS)
    
    all_results = []
    
    for trial_idx in range(NUM_TRIALS):
        seed = trial_seeds[trial_idx]
        trial_df = run_single_trial(
            trial_idx, seed, rdc_agent, uic_agents,
            dropoff_demand_forecaster, pickup_demand_forecaster,
            dropoff_demand_provider, pickup_demand_provider
        )
        all_results.append(trial_df)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_results_path = os.path.join(OUTPUT_DIR, f"hrl_evaluation_raw_results_{timestamp}.pickle")
    combined_results.to_pickle(raw_results_path)
    
    csv_results_path = os.path.join(OUTPUT_DIR, f"hrl_evaluation_raw_results_{timestamp}.csv")
    combined_results.to_csv(csv_results_path, index=False)
    
    print(f"\n{'='*80}")
    print("CALCULATING SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    metadata_cols = ['trial', 'seed', 'configuration', 'manual_rebalancing', 
                    'incentive_rebalancing', 'fleet_size']
    numeric_cols = [col for col in combined_results.columns 
                   if col not in metadata_cols and pd.api.types.is_numeric_dtype(combined_results[col])]
    
    summary_stats = combined_results.groupby('configuration')[numeric_cols].agg(['mean', 'std']).round(4)
    
    summary_stats.columns = [f'{col[0]}_{col[1]}' for col in summary_stats.columns]
    
    config_metadata = combined_results.groupby('configuration')[metadata_cols[2:]].first()
    summary_results = pd.concat([config_metadata, summary_stats], axis=1)
    
    summary_results_path = os.path.join(OUTPUT_DIR, f"hrl_evaluation_summary_results_{timestamp}.pickle")
    summary_results.to_pickle(summary_results_path)
    
    summary_csv_path = os.path.join(OUTPUT_DIR, f"hrl_evaluation_summary_results_{timestamp}.csv")
    summary_results.to_csv(summary_csv_path)
    
    # Print summary
    print(f"\nSUMMARY RESULTS (Mean ± Std across {NUM_TRIALS} trials)")
    print(f"{'-'*120}")
    print(f"{'Configuration':<25} {'Fleet':<6} {'Mean Satisfied':<20} {'Manual Rebal':<20} {'Incentive Rebal':<20} {'Gini Index':<20}")
    print(f"{'-'*120}")
    
    for config in summary_results.index:
        row = summary_results.loc[config]
        satisfied_mean = row.get('mean_satisfied_ratio_mean', 0)
        satisfied_std = row.get('mean_satisfied_ratio_std', 0)
        manual_mean = row.get('mean_rebalanced_vehicles_manually_mean', 0)
        manual_std = row.get('mean_rebalanced_vehicles_manually_std', 0)
        incentive_mean = row.get('mean_rebalanced_vehicles_incentives_mean', 0)
        incentive_std = row.get('mean_rebalanced_vehicles_incentives_std', 0)
        gini_mean = row.get('mean_gini_index_mean', 0)
        gini_std = row.get('mean_gini_index_std', 0)
        
        print(f"{config:<25} {row['fleet_size']:<6.0f} "
              f"{satisfied_mean:.3f}±{satisfied_std:.3f}        "
              f"{manual_mean:.1f}±{manual_std:.1f}          "
              f"{incentive_mean:.1f}±{incentive_std:.1f}            "
              f"{gini_mean:.3f}±{gini_std:.3f}")
        
    print("HRL framework multi-trial evaluation completed!")


def main():
    """Main function to run the evaluation."""
    run_evaluation()


if __name__ == "__main__":
    main()