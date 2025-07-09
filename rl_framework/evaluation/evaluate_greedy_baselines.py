"""
evaluate_greedy_baselines.py

Main evaluation script for greedy heuristic baselines.
This script evaluates simple greedy strategies for both manual rebalancing
and user incentive coordination to serve as baselines for the HRL framework.
"""

import sys
import os

# Add the parent directories to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demand_forecasting.IrConv_LSTM_pre_forecaster import IrConvLstmDemandPreForecaster
from demand_provider.demand_provider_impl import DemandProviderImpl
from greedy_heuristics import (
    GreedyManualRebalancer,
    GreedyIncentiveCoordinator,
    GreedyHeuristicEvaluator,
)
import pandas as pd
from datetime import timedelta, datetime
from uic_training_loop import USER_WILLINGNESS_FN
from training_loop import RDC_ACTION_VALUES

# Configuration
START_TIME = datetime(2025, 5, 18, 15, 0)
END_TIME = datetime(2025, 6, 18, 15, 0)
STEP_DURATION = 60
MAX_STEPS = 400

# Test different combinations
CONFIGURATIONS = [
    {
        "manual": True,
        "incentive": True,
        "name": "Both_Enabled_900",
        "fleet_size": 900,
    },
    {
        "manual": True,
        "incentive": False,
        "name": "Manual_Only_900",
        "fleet_size": 900,
    },
    {
        "manual": False,
        "incentive": True,
        "name": "Incentive_Only_900",
        "fleet_size": 900,
    },
    {
        "manual": False,
        "incentive": False,
        "name": "No_Rebalancing_900",
        "fleet_size": 900,
    },
    {
        "manual": True,
        "incentive": True,
        "name": "Both_Enabled_600",
        "fleet_size": 600,
    },
    {
        "manual": True,
        "incentive": False,
        "name": "Manual_Only_600",
        "fleet_size": 600,
    },
    {
        "manual": False,
        "incentive": True,
        "name": "Incentive_Only_600",
        "fleet_size": 600,
    },
    {
        "manual": False,
        "incentive": False,
        "name": "No_Rebalancing_600",
        "fleet_size": 600,
    },
    {
        "manual": True,
        "incentive": True,
        "name": "Both_Enabled_1200",
        "fleet_size": 1200,
    },
    {
        "manual": True,
        "incentive": False,
        "name": "Manual_Only_1200",
        "fleet_size": 1200,
    },
    {
        "manual": False,
        "incentive": True,
        "name": "Incentive_Only_1200",
        "fleet_size": 1200,
    },
    {
        "manual": False,
        "incentive": False,
        "name": "No_Rebalancing_1200",
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


def run_greedy_evaluation() -> None:
    """Run evaluation of greedy heuristic baselines.

    Tests different combinations of manual and incentive-based rebalancing
    using simple greedy strategies to establish baseline performance.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
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

    manual_rebalancer = GreedyManualRebalancer(
        action_values=RDC_ACTION_VALUES,
        num_communities=NUM_COMMUNITIES,
        pickup_demand_forecaster=pickup_demand_forecaster,
        max_rebalancing_ratio=0.10,
    )

    incentive_coordinators = []
    for _ in range(NUM_COMMUNITIES):
        coordinator = GreedyIncentiveCoordinator(
            pickup_demand_forecaster=pickup_demand_forecaster,
            dropoff_demand_forecaster=dropoff_demand_forecaster,
            incentive_threshold_ratio=0.75,
        )
        incentive_coordinators.append(coordinator)

    results = {}

    for config in CONFIGURATIONS:
        print(f"\n{'='*60}")
        print(f"Evaluating Configuration: {config['name']}")
        print(f"Manual Rebalancing: {config['manual']}")
        print(f"Incentive Rebalancing: {config['incentive']}")
        print(f"Fleet Size: {config['fleet_size']}")
        print(f"{'='*60}")

        evaluator = GreedyHeuristicEvaluator(
            manual_rebalancer=manual_rebalancer,
            incentive_coordinators=incentive_coordinators,
            zone_community_map=ZONE_COMMUNITY_MAP,
            community_index_map=COMMUNITY_INDEX_MAP,
            zone_index_map=ZONE_INDEX_MAP,
            pickup_demand_forecaster=pickup_demand_forecaster,
            dropoff_demand_forecaster=dropoff_demand_forecaster,
            pickup_demand_provider=pickup_demand_provider,
            dropoff_demand_provider=dropoff_demand_provider,
            fleet_size=config["fleet_size"],
            start_time=START_TIME,
            max_steps=MAX_STEPS,
            step_duration=timedelta(minutes=STEP_DURATION),
            user_willingness_fn=USER_WILLINGNESS_FN,
            zone_neighbor_map=ZONE_NEIGHBOR_MAP,
            enable_manual_rebalancing=config["manual"],
            enable_incentive_rebalancing=config["incentive"],
        )

        config_results = evaluator.evaluate()
        results[config["name"]] = config_results

        print(f"\nResults for {config['name']}:")
        for key, value in config_results.items():
            print(f"  {key}: {value:.4f}")

    # Print summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(
        f"{'Configuration':<20} {'Mean Satisfied':<15} {'Manual Rebal':<15} {'Incentive Rebal':<18} {'Fleet Size':<10}"
    )
    print(f"{'-'*80}")

    for config_name, config_results in results.items():
        print(
            f"{config_name:<20} "
            f"{config_results['mean_satisfied_ratio']:<15.4f} "
            f"{config_results['mean_rebalanced_vehicles_manually']:<15.1f} "
            f"{config_results['mean_rebalanced_vehicles_incentives']:<18.1f}"
            f"{config_results['fleet_size']:<10d}"
        )

    print("\nGreedy baseline evaluation completed!")


if __name__ == "__main__":
    run_greedy_evaluation()
