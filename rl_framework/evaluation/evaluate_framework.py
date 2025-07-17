"""
evaluate_framework.py

Main evaluation script for the Hierarchical Reinforcement Learning framework.
This script loads trained RDC and UIC agents and evaluates their coordinated performance
in managing e-scooter fleet distribution across multiple communities.
"""

import sys
import os

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

# Mean in GBFS data is around 900 scooters
# [750, 900, 1200]
START_TIME = datetime(2025, 5, 18, 15, 0)
END_TIME = datetime(2025, 6, 18, 15, 0)
STEP_DURATION = 60
MAX_STEPS = 400
RDC_AGENT_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/runs/outputs/rdc_agent_model_20250713-220126.pth"
# RDC_AGENT_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/runs/outputs/rdc_agent_model_20250703-173727.pth"
UIC_AGENT_PATHS = [
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa44fffffff_uic_model_20250717-155130.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa637ffffff_uic_model_20250717-155132.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa707ffffff_uic_model_20250717-155130.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa717ffffff_uic_model_20250717-170106.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa71fffffff_uic_model_20250717-170039.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa787ffffff_uic_model_20250717-165959.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa78fffffff_uic_model_20250717-180816.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa7a7ffffff_uic_model_20250717-180821.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa7afffffff_uic_model_20250717-181051.zip",
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
        "manual": False,
        "incentive": False,
        "name": "HRL_No_Rebalancing_900",
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
        "manual": False,
        "incentive": False,
        "name": "HRL_No_Rebalancing_600",
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
    {
        "manual": False,
        "incentive": False,
        "name": "HRL_No_Rebalancing_1200",
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


def run_evaluation() -> None:
    """Run complete evaluation of the HRL framework.

    Loads trained RDC and UIC agents, sets up the evaluation environment,
    and measures the coordinated performance of both agent types working together
    to manage e-scooter fleet distribution and user incentives.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
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

    results_summary = {}

    for config in CONFIGURATIONS:
        print(f"\n{'='*60}")
        print(f"Evaluating Configuration: {config['name']}")
        print(f"Manual Rebalancing: {config['manual']}")
        print(f"Incentive Rebalancing: {config['incentive']}")
        print(f"Fleet Size: {config['fleet_size']}")
        print(f"{'='*60}")

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
        results["fleet_size"] = config["fleet_size"]
        results_summary[config["name"]] = results

        print(f"\nResults for {config['name']}:")
        for key, value in results.items():
            print(f"  {key}: {value}")

    # Print summary comparison
    print(f"\n{'='*80}")
    print("HRL FRAMEWORK - SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(
        f"{'Configuration':<25} {'Mean Satisfied':<15} {'Manual Rebal':<15} {'Incentive Rebal':<18} {'Mean Gini Index':<15} {'Fleet Size':<10}"
    )
    print(f"{'-'*80}")

    for config_name, config_results in results_summary.items():
        print(
            f"{config_name:<25} "
            f"{config_results['mean_satisfied_ratio']:<15.4f} "
            f"{config_results['mean_rebalanced_vehicles_manually']:<15.1f} "
            f"{config_results['mean_rebalanced_vehicles_incentives']:<18.1f}"
            f"{config_results['mean_gini_index']:<15.4f} "
            f"{config_results['fleet_size']:<10d}"
        )

    print("\nHRL framework evaluation completed!")


if __name__ == "__main__":
    run_evaluation()
