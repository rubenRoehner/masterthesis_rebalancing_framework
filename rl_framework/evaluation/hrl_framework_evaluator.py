"""
hrl_framework_evaluator.py

Hierarchical Reinforcement Learning Framework Evaluator.
This module provides evaluation capabilities for the complete HRL framework,
coordinating both Regional Distribution Coordinator (RDC) and User Incentive Coordinator (UIC) agents
to assess overall system performance in e-scooter fleet management.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Callable
from tqdm import tqdm
from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider import DemandProvider
from regional_distribution_coordinator.regional_distribution_coordinator import (
    RegionalDistributionCoordinator,
)
from regional_distribution_coordinator.escooter_rdc_env import EscooterRDCEnv
from stable_baselines3 import PPO
from user_incentive_coordinator.escooter_uic_env import EscooterUICEnv


class HRLFrameworkEvaluator:
    """Evaluator for the Hierarchical Reinforcement Learning framework.

    This class coordinates the evaluation of both RDC and UIC agents working together
    to manage e-scooter fleet distribution across multiple communities, measuring
    overall system performance and demand satisfaction.
    """

    def __init__(
        self,
        rdc_agent: RegionalDistributionCoordinator,
        uic_agents: list[PPO],
        zone_community_map: pd.DataFrame,
        community_index_map: dict[str, int],
        zone_index_map: dict[str, int],
        pickup_demand_forecaster: DemandForecaster,
        dropoff_demand_forecaster: DemandForecaster,
        pickup_demand_provider: DemandProvider,
        dropoff_demand_provider: DemandProvider,
        fleet_size: int,
        start_time: datetime,
        max_steps: int,
        step_duration: timedelta,
        user_willingness_fn: Callable[[float], float],
        zone_neighbor_map: dict[str, list[str]],
        device: torch.device,
        enable_rdc_rebalancing: bool = True,
        enable_uic_rebalancing: bool = True,
    ) -> None:
        """Initialize the HRL Framework Evaluator.

        Args:
            rdc_agent: trained Regional Distribution Coordinator agent
            uic_agents: list of trained User Incentive Coordinator agents for each community
            zone_community_map: DataFrame mapping zones to communities
            community_index_map: mapping from community IDs to indices
            zone_index_map: mapping from zone IDs to indices
            pickup_demand_forecaster: forecaster for pickup demand patterns
            dropoff_demand_forecaster: forecaster for dropoff demand patterns
            pickup_demand_provider: provider for actual pickup demand data
            dropoff_demand_provider: provider for actual dropoff demand data
            fleet_size: total number of vehicles in the fleet
            start_time: evaluation start time
            max_steps: maximum number of evaluation steps
            step_duration: duration of each evaluation step
            user_willingness_fn: function mapping incentive to user compliance probability
            zone_neighbor_map: mapping from zone IDs to neighbor zone IDs
            device: PyTorch device for tensor operations
            enable_rdc_rebalancing: whether to enable RDC manual rebalancing
            enable_uic_rebalancing: whether to enable UIC incentive-based rebalancing

        Returns:
            None

        Raises:
            None
        """
        self.rdc_agent = rdc_agent
        self.uic_agents = uic_agents
        self.zone_community_map = zone_community_map
        self.community_index_map = community_index_map
        self.zone_index_map = zone_index_map
        self.pickup_demand_forecaster = pickup_demand_forecaster
        self.dropoff_demand_forecaster = dropoff_demand_forecaster
        self.pickup_demand_provider = pickup_demand_provider
        self.dropoff_demand_provider = dropoff_demand_provider
        self.fleet_size = fleet_size
        self.start_time = start_time
        self.max_steps = max_steps
        self.user_willingness_fn = user_willingness_fn

        self.zone_neighbor_map = zone_neighbor_map
        self.n_communities = len(community_index_map)
        self.n_total_zones = zone_community_map.shape[0]

        self.zones_in_community: Dict[str, List[int]] = {}
        self.community_zone_index_maps: Dict[str, Dict[str, int]] = {}
        self.community_zone_neighbor_maps: Dict[str, Dict[str, List[str]]] = {}

        for community_id in community_index_map.keys():
            community_zones_df = zone_community_map[
                zone_community_map["community_index"] == community_id
            ]
            zone_ids = list(community_zones_df["grid_index"])

            community_zone_index_map = {}
            for local_idx, (i, row) in enumerate(community_zones_df.iterrows()):
                community_zone_index_map[row["grid_index"]] = local_idx

            community_zone_ids_set = set(zone_ids)
            community_zone_neighbor_map = {}
            for zone_id in zone_ids:
                if zone_id in zone_neighbor_map:
                    neighbors = [
                        neighbor
                        for neighbor in zone_neighbor_map[zone_id]
                        if neighbor in community_zone_ids_set
                    ]
                    community_zone_neighbor_map[zone_id] = neighbors
                else:
                    community_zone_neighbor_map[zone_id] = []

            global_zone_indices = [zone_index_map[zone_id] for zone_id in zone_ids]

            self.zones_in_community[community_id] = global_zone_indices
            self.community_zone_index_maps[community_id] = community_zone_index_map
            self.community_zone_neighbor_maps[community_id] = (
                community_zone_neighbor_map
            )

        self.global_vehicle_counts = np.full(
            self.n_total_zones, self.fleet_size // self.n_total_zones, dtype=int
        )
        remainder = fleet_size % self.n_total_zones
        for i in range(remainder):
            self.global_vehicle_counts[i] += 1

        self.current_time = self.start_time
        self.max_steps = max_steps
        self.step_duration = step_duration
        self.current_step = 0

        self.rdc_cumulative_reward = 0.0
        self.uic_cumulative_rewards = [0.0 for _ in range(self.n_communities)]

        self.device = device
        self.enable_rdc_rebalancing = enable_rdc_rebalancing
        self.enable_uic_rebalancing = enable_uic_rebalancing

    def get_vehicle_counts_per_community(self) -> np.ndarray:
        """Get the total number of vehicles in each community.

        Args:
            None

        Returns:
            np.ndarray: array of vehicle counts per community

        Raises:
            None
        """
        vehicle_counts_per_community = np.zeros(self.n_communities, dtype=int)
        for community_id, community_index in self.community_index_map.items():
            global_zone_indices = self.zones_in_community[community_id]
            vehicle_counts_per_community[community_index] = self.global_vehicle_counts[
                global_zone_indices
            ].sum()
        return vehicle_counts_per_community

    def get_rdc_state_observation(self) -> torch.Tensor:
        """Get the current state observation for the RDC agent.

        Args:
            None

        Returns:
            torch.Tensor: state observation for regional distribution coordination

        Raises:
            None
        """
        observation = []
        current_pickup_demand_forecast = (
            self.pickup_demand_forecaster.predict_demand_per_community(
                time_of_day=self.current_time.hour,
                day=self.current_time.day,
                month=self.current_time.month,
            )
        )
        current_dropoff_demand_forecast = (
            self.dropoff_demand_forecaster.predict_demand_per_community(
                time_of_day=self.current_time.hour,
                day=self.current_time.day,
                month=self.current_time.month,
            )
        )
        vehicle_counts_per_community = self.get_vehicle_counts_per_community()

        for i in range(self.n_communities):
            community_observation = [
                current_pickup_demand_forecast[i],
                current_dropoff_demand_forecast[i],
                vehicle_counts_per_community[i],
            ]
            observation.extend(community_observation)

        return torch.tensor(observation, dtype=torch.float32, device=self.device)

    def get_uic_state_observation(self, community_id: str) -> Dict[str, np.ndarray]:
        """Get the current state observation for a specific UIC agent.

        Args:
            community_id: ID of the community for UIC observation

        Returns:
            Dict[str, np.ndarray]: state observation for user incentive coordination

        Raises:
            None
        """
        global_zone_indices = self.zones_in_community[community_id]
        vehicle_counts = (
            self.global_vehicle_counts[global_zone_indices].astype(np.int32).copy()
        )

        pickup_forecast_full = self.pickup_demand_forecaster.predict_demand_per_zone(
            time_of_day=self.current_time.hour,
            day=self.current_time.day,
            month=self.current_time.month,
        )
        dropoff_forecast_full = self.dropoff_demand_forecaster.predict_demand_per_zone(
            time_of_day=self.current_time.hour,
            day=self.current_time.day,
            month=self.current_time.month,
        )

        pickup_demand_forecast = pickup_forecast_full[global_zone_indices].astype(
            np.float32
        )
        dropoff_demand_forecast = dropoff_forecast_full[global_zone_indices].astype(
            np.float32
        )

        return {
            "pickup_demand": pickup_demand_forecast,
            "dropoff_demand": dropoff_demand_forecast,
            "current_vehicle_counts": vehicle_counts,
        }

    def evaluate(self) -> Dict:
        """Evaluate the complete HRL framework performance.

        Runs the evaluation episode with both RDC and UIC agents working together,
        measuring demand satisfaction and overall system performance.

        Args:
            None

        Returns:
            Dict: evaluation results including mean satisfied demand ratio

        Raises:
            None
        """
        satisfied_ratio = []
        total_rebalanced_vehicles_manually = []
        total_rebalanced_vehicles_incentives = []
        gini_index = []

        with tqdm(
            total=self.max_steps, desc="Evaluating HRL Framework", unit="step"
        ) as pbar:
            while self.current_step < self.max_steps:
                # --- RDC operator based rebalancing ---
                obs_rdc = self.get_rdc_state_observation()

                with torch.no_grad():
                    rdc_action_inidces = self.rdc_agent.select_action(obs_rdc)

                rdc_action_inidces = [int(a) for a in rdc_action_inidces]

                rdc_allocations = [
                    self.rdc_agent.action_values[a] for a in rdc_action_inidces
                ]

                vehicles_rebalanced_manually = 0
                if self.enable_rdc_rebalancing:
                    vehicles_rebalanced_manually, self.global_vehicle_counts = (
                        EscooterRDCEnv.handle_rebalancing(
                            actions=rdc_allocations,
                            current_vehicle_counts=self.global_vehicle_counts,
                            community_index_map=self.community_index_map,
                            zone_community_map=self.zone_community_map,
                            zone_index_map=self.zone_index_map,
                        )
                    )

                satisfied_per_community = [0 for _ in range(self.n_communities)]
                offered_per_community = [0 for _ in range(self.n_communities)]
                vehicles_rebalanced_uic = [0 for _ in range(self.n_communities)]

                for community_id, community_index in self.community_index_map.items():
                    uic_observation = self.get_uic_state_observation(
                        community_id=community_id
                    )

                    uic_action, _ = self.uic_agents[community_index].predict(
                        uic_observation, deterministic=True
                    )

                    local_dropoff_demand = (
                        self.dropoff_demand_provider.get_demand_per_zone_community(
                            time_of_day=self.current_time.hour,
                            day=self.current_time.day,
                            month=self.current_time.month,
                            community_id=community_id,
                        )
                    )
                    local_pickup_demand = (
                        self.pickup_demand_provider.get_demand_per_zone_community(
                            time_of_day=self.current_time.hour,
                            day=self.current_time.day,
                            month=self.current_time.month,
                            community_id=community_id,
                        )
                    )
                    offered_demand = local_pickup_demand.copy().sum()

                    global_zone_indices = self.zones_in_community[community_id]

                    zone_community_df = self.zone_community_map[
                        self.zone_community_map["community_index"] == community_id
                    ]
                    zone_ids = list(zone_community_df["grid_index"])
                    zone_id_to_local = self.community_zone_index_maps[community_id]
                    zone_neighbor_map_local = self.community_zone_neighbor_maps[
                        community_id
                    ]

                    local_vehicle_counts = (
                        self.global_vehicle_counts[global_zone_indices]
                        .astype(int)
                        .copy()
                    )
                    if self.enable_uic_rebalancing:
                        (
                            local_dropoff_demand,
                            vehicles_rebalanced_uic[community_index],
                        ) = EscooterUICEnv.handle_incentives(
                            action=uic_action,
                            dropoff_demand=local_dropoff_demand,
                            zone_ids=zone_ids,
                            zone_id_to_local=zone_id_to_local,
                            zone_neighbor_map=zone_neighbor_map_local,
                            user_willingness_fn=self.user_willingness_fn,
                        )
                    else:
                        vehicles_rebalanced_uic[community_index] = 0

                    local_vehicle_counts, total_satisfied_demand = (
                        EscooterUICEnv.update_vehicle_counts(
                            n_zones=len(local_vehicle_counts),
                            pickup_demand=local_pickup_demand,
                            dropoff_demand=local_dropoff_demand,
                            current_vehicle_counts=local_vehicle_counts,
                        )
                    )

                    # Update global vehicle counts
                    self.global_vehicle_counts[global_zone_indices] = (
                        local_vehicle_counts
                    )
                    if offered_demand > 0:
                        satisfied_per_community[community_index] = (
                            total_satisfied_demand
                        )
                        offered_per_community[community_index] = offered_demand

                total_satisfied_all = sum(satisfied_per_community)
                total_offered_all = sum(offered_per_community)
                total_rebalanced_vehicles_incentives.append(
                    sum(vehicles_rebalanced_uic)
                )
                total_rebalanced_vehicles_manually.append(vehicles_rebalanced_manually)
                if total_offered_all > 0:
                    satisfied_ratio_global = total_satisfied_all / float(
                        total_offered_all
                    )
                else:
                    satisfied_ratio_global = 1.0

                assert (
                    self.global_vehicle_counts.sum() == self.fleet_size
                ), "Vehicle counts do not match the total fleet size after demand simulation."

                satisfied_ratio.append(satisfied_ratio_global)
                self.current_step += 1
                self.current_time += self.step_duration

                gini_index.append(
                    EscooterRDCEnv.calculate_gini_coefficient(
                        self.global_vehicle_counts.copy()
                    )
                )

                # Update progress bar with current metrics
                pbar.set_postfix(
                    {
                        "satisfied_ratio": f"{satisfied_ratio_global:.3f}",
                        "manual_rebal": vehicles_rebalanced_manually,
                        "incentive_rebal": sum(vehicles_rebalanced_uic),
                    }
                )
                pbar.update(1)

        return {
            "mean_satisfied_ratio": np.mean(satisfied_ratio),
            "mean_rebalanced_vehicles_manually": np.mean(
                total_rebalanced_vehicles_manually
            ),
            "mean_rebalanced_vehicles_incentives": np.mean(
                total_rebalanced_vehicles_incentives
            ),
            "max_satisfied_ratio": np.max(satisfied_ratio),
            "min_satisfied_ratio": np.min(satisfied_ratio),
            "mean_gini_index": np.mean(gini_index),
        }
