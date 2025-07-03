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

FLEET_SIZE = 600
START_TIME = datetime(2025, 5, 18, 15, 0)
END_TIME = datetime(2025, 6, 18, 15, 0)
STEP_DURATION = 60
MAX_STEPS = 400
ENABLE_RDC_REBALANCING = True
ENABLE_UIC_REBALANCING = True
RDC_AGENT_PATH = "/home/ruroit00/rebalancing_framework/rl_framework/runs/outputs/rdc_agent_model_20250630-184804.pth"
UIC_AGENT_PATHS = [
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa44fffffff_user_incentive_coordinator.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa637ffffff_user_incentive_coordinator.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa707ffffff_user_incentive_coordinator.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa717ffffff_user_incentive_coordinator.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa71fffffff_user_incentive_coordinator.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa787ffffff_user_incentive_coordinator.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa78fffffff_user_incentive_coordinator.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa7a7ffffff_user_incentive_coordinator.zip",
    "/home/ruroit00/rebalancing_framework/rl_framework/runs/UIC/outputs/861faa7afffffff_user_incentive_coordinator.zip",
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


torch.cuda.set_device(2)
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

    evaluator = HRLFrameworkEvaluator(
        fleet_size=FLEET_SIZE,
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
        enable_rdc_rebalancing=ENABLE_RDC_REBALANCING,
        enable_uic_rebalancing=ENABLE_UIC_REBALANCING,
    )

    results = evaluator.evaluate()
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_evaluation()
