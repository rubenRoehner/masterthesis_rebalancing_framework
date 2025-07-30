"""
greedy_heuristics.py

Greedy heuristic implementations for e-scooter rebalancing.
This module provides baseline algorithms for both manual rebalancing (RDC-style)
and user incentive coordination (UIC-style) using simple greedy strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from datetime import datetime
from tqdm import tqdm

from regional_distribution_coordinator.escooter_rdc_env import EscooterRDCEnv
from user_incentive_coordinator.escooter_uic_env import EscooterUICEnv
from demand_forecasting.demand_forecaster import DemandForecaster


class GreedyManualRebalancer:
    """Greedy heuristic for manual vehicle rebalancing across communities.

    This heuristic moves vehicles from communities with excess supply to those
    with high demand, similar to the RDC but using a simple greedy strategy.
    """

    def __init__(
        self,
        action_values: List[int],
        num_communities: int,
        pickup_demand_forecaster: DemandForecaster,
        max_rebalancing_ratio: float = 0.2,
    ) -> None:
        """Initialize the greedy manual rebalancer.

        Args:
            action_values: possible rebalancing action values
            num_communities: number of communities to coordinate
            pickup_demand_forecaster: forecaster for pickup demand
            max_rebalancing_ratio: maximum fraction of fleet to rebalance per step

        Returns:
            None

        Raises:
            None
        """
        self.action_values = action_values
        self.num_communities = num_communities
        self.pickup_demand_forecaster = pickup_demand_forecaster
        self.max_rebalancing_ratio = max_rebalancing_ratio

    def select_action(
        self,
        vehicle_counts_per_community: np.ndarray,
        current_time: datetime,
        fleet_size: int,
    ) -> List[int]:
        """Select rebalancing actions using greedy strategy.

        Strategy: Move vehicles from communities with surplus (vehicles > demand)
        to communities with deficit (demand > vehicles), prioritizing by deficit size.

        Args:
            vehicle_counts_per_community: current vehicle counts per community
            current_time: current simulation time
            fleet_size: total fleet size

        Returns:
            List[int]: action indices for each community

        Raises:
            None
        """
        demand_forecast = self.pickup_demand_forecaster.predict_demand_per_community(
            time_of_day=current_time.hour,
            day=current_time.day,
            month=current_time.month,
        )

        deficit = demand_forecast - vehicle_counts_per_community

        max_total_rebalancing = int(fleet_size * self.max_rebalancing_ratio)

        actions = [0] * self.num_communities

        deficit_sorted_indices = np.argsort(-deficit)

        total_rebalanced = 0

        max_positive_action = max([v for v in self.action_values if v > 0], default=0)

        for i in deficit_sorted_indices:
            if total_rebalanced >= max_total_rebalancing:
                break

            community_deficit = deficit[i]

            if community_deficit > 0:
                vehicles_needed = int(
                    min(
                        float(community_deficit),
                        max_total_rebalancing - total_rebalanced,
                        max_positive_action,
                    )
                )

                if vehicles_needed > 0:
                    best_action_index = 0
                    best_action_value = 0
                    for index, value in enumerate(self.action_values):
                        if 0 < value <= vehicles_needed and value > best_action_value:
                            best_action_index = index
                            best_action_value = value

                    if best_action_value > 0:
                        actions[i] = best_action_index
                        total_rebalanced += best_action_value

        max_negative_action = abs(
            min([v for v in self.action_values if v < 0], default=0)
        )

        for i in deficit_sorted_indices[::-1]:
            if total_rebalanced >= max_total_rebalancing:
                break

            community_deficit = deficit[i]

            if community_deficit < 0 and vehicle_counts_per_community[i] > 0:
                vehicles_to_remove = int(
                    min(
                        float(-community_deficit),
                        max_total_rebalancing - total_rebalanced,
                        max_negative_action,
                        float(vehicle_counts_per_community[i]),
                    )
                )

                if vehicles_to_remove > 0:
                    best_action_index = 0
                    best_action_value = 0
                    for index, value in enumerate(self.action_values):
                        if (
                            value < 0
                            and -value <= vehicles_to_remove
                            and -value > best_action_value
                        ):
                            best_action_index = index
                            best_action_value = -value

                    if best_action_value > 0:
                        actions[i] = best_action_index
                        total_rebalanced += best_action_value

        return actions


class GreedyIncentiveCoordinator:
    """Greedy heuristic for user incentive coordination within a community.

    This heuristic provides maximum incentives to zones that need more vehicles
    and no incentives to zones that have sufficient or excess vehicles.
    """

    def __init__(
        self,
        pickup_demand_forecaster: DemandForecaster,
        dropoff_demand_forecaster: DemandForecaster,
        incentive_threshold_ratio: float = 0.8,
    ) -> None:
        """Initialize the greedy incentive coordinator.

        Args:
            pickup_demand_forecaster: forecaster for pickup demand
            dropoff_demand_forecaster: forecaster for dropoff demand
            incentive_threshold_ratio: ratio below which zones get max incentives

        Returns:
            None

        Raises:
            None
        """
        self.pickup_demand_forecaster = pickup_demand_forecaster
        self.dropoff_demand_forecaster = dropoff_demand_forecaster
        self.incentive_threshold_ratio = incentive_threshold_ratio

    def select_action(
        self,
        vehicle_counts: np.ndarray,
        current_time: datetime,
        zone_indices: List[int],
    ) -> np.ndarray:
        """Select incentive levels using greedy strategy.

        Strategy: Provide maximum incentives (1.0) to zones where vehicle count
        is below threshold relative to expected demand, zero incentives otherwise.

        Args:
            community_id: ID of the community
            vehicle_counts: current vehicle counts per zone in community
            current_time: current simulation time
            zone_indices: global zone indices for this community

        Returns:
            np.ndarray: incentive levels for each zone (0-1)

        Raises:
            None
        """
        pickup_forecast_full = self.pickup_demand_forecaster.predict_demand_per_zone(
            time_of_day=current_time.hour,
            day=current_time.day,
            month=current_time.month,
        )

        pickup_forecast = pickup_forecast_full[zone_indices]

        incentives = np.zeros(len(vehicle_counts), dtype=np.float32)

        for i in range(len(vehicle_counts)):
            expected_demand = pickup_forecast[i]
            current_supply = vehicle_counts[i]

            if expected_demand > 0:
                supply_ratio = current_supply / expected_demand

                if supply_ratio < self.incentive_threshold_ratio:
                    incentives[i] = 1.0
                else:
                    incentives[i] = 0.0
            else:
                incentives[i] = 0.0

        return incentives


class GreedyHeuristicEvaluator:
    """Evaluator for greedy heuristic baselines.

    This class evaluates greedy heuristics using the same framework as the HRL evaluation,
    allowing direct comparison of performance metrics.
    """

    def __init__(
        self,
        manual_rebalancer: GreedyManualRebalancer,
        incentive_coordinators: List[GreedyIncentiveCoordinator],
        zone_community_map: pd.DataFrame,
        community_index_map: Dict[str, int],
        zone_index_map: Dict[str, int],
        pickup_demand_forecaster: DemandForecaster,
        dropoff_demand_forecaster: DemandForecaster,
        pickup_demand_provider,
        dropoff_demand_provider,
        fleet_size: int,
        start_time: datetime,
        max_steps: int,
        step_duration,
        user_willingness_fn: Callable[[float], float],
        zone_neighbor_map: Dict[str, List[str]],
        enable_manual_rebalancing: bool = True,
        enable_incentive_rebalancing: bool = True,
    ) -> None:
        """Initialize the greedy heuristic evaluator.

        Args:
            manual_rebalancer: greedy manual rebalancing heuristic
            incentive_coordinators: list of greedy incentive coordinators
            zone_community_map: DataFrame mapping zones to communities
            community_index_map: mapping from community IDs to indices
            zone_index_map: mapping from zone IDs to indices
            pickup_demand_forecaster: forecaster for pickup demand
            dropoff_demand_forecaster: forecaster for dropoff demand
            pickup_demand_provider: provider for actual pickup demand
            dropoff_demand_provider: provider for actual dropoff demand
            fleet_size: total fleet size
            start_time: evaluation start time
            max_steps: maximum evaluation steps
            step_duration: duration of each step
            user_willingness_fn: function mapping incentive to compliance
            zone_neighbor_map: mapping from zone IDs to neighbors
            enable_manual_rebalancing: whether to enable manual rebalancing
            enable_incentive_rebalancing: whether to enable incentive rebalancing

        Returns:
            None

        Raises:
            None
        """
        self.manual_rebalancer = manual_rebalancer
        self.incentive_coordinators = incentive_coordinators
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
        self.step_duration = step_duration
        self.user_willingness_fn = user_willingness_fn
        self.zone_neighbor_map = zone_neighbor_map
        self.enable_manual_rebalancing = enable_manual_rebalancing
        self.enable_incentive_rebalancing = enable_incentive_rebalancing

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
        self.current_step = 0

    def get_vehicle_counts_per_community(self) -> np.ndarray:
        """Get vehicle counts per community."""
        vehicle_counts_per_community = np.zeros(self.n_communities, dtype=int)
        for community_id, community_index in self.community_index_map.items():
            global_zone_indices = self.zones_in_community[community_id]
            vehicle_counts_per_community[community_index] = self.global_vehicle_counts[
                global_zone_indices
            ].sum()
        return vehicle_counts_per_community

    def evaluate(self) -> Dict:
        """Evaluate the greedy heuristics performance.

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
            total=self.max_steps, desc="Evaluating Greedy Heuristics", unit="step"
        ) as pbar:
            while self.current_step < self.max_steps:
                vehicle_counts_per_community = self.get_vehicle_counts_per_community()

                manual_actions = self.manual_rebalancer.select_action(
                    vehicle_counts_per_community=vehicle_counts_per_community,
                    current_time=self.current_time,
                    fleet_size=self.fleet_size,
                )

                manual_allocations = [
                    self.manual_rebalancer.action_values[action_index]
                    for action_index in manual_actions
                ]

                vehicles_rebalanced_manually = 0
                if self.enable_manual_rebalancing:
                    vehicles_rebalanced_manually, self.global_vehicle_counts = (
                        EscooterRDCEnv.handle_rebalancing(
                            actions=manual_allocations,
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
                    global_zone_indices = self.zones_in_community[community_id]
                    local_vehicle_counts = (
                        self.global_vehicle_counts[global_zone_indices]
                        .astype(int)
                        .copy()
                    )

                    incentive_action = self.incentive_coordinators[
                        community_index
                    ].select_action(
                        vehicle_counts=local_vehicle_counts,
                        current_time=self.current_time,
                        zone_indices=global_zone_indices,
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
                    offered_demand = local_pickup_demand.sum()

                    zone_community_df = self.zone_community_map[
                        self.zone_community_map["community_index"] == community_id
                    ]
                    zone_ids = list(zone_community_df["grid_index"])
                    zone_id_to_local = self.community_zone_index_maps[community_id]
                    zone_neighbor_map_local = self.community_zone_neighbor_maps[
                        community_id
                    ]

                    if self.enable_incentive_rebalancing:
                        (
                            local_dropoff_demand,
                            vehicles_rebalanced_uic[community_index],
                        ) = EscooterUICEnv.handle_incentives(
                            action=incentive_action,
                            dropoff_demand=local_dropoff_demand,
                            zone_ids=zone_ids,
                            zone_id_to_local=zone_id_to_local,
                            zone_neighbor_map=zone_neighbor_map_local,
                            user_willingness_fn=self.user_willingness_fn,
                        )
                    else:
                        vehicles_rebalanced_uic[community_index] = 0

                    local_vehicle_counts, total_satisfied_demand, offered_demand = (
                        EscooterUICEnv.update_vehicle_counts(
                            n_zones=len(local_vehicle_counts),
                            pickup_demand=local_pickup_demand,
                            dropoff_demand=local_dropoff_demand,
                            current_vehicle_counts=local_vehicle_counts,
                        )
                    )

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

                satisfied_ratio.append(satisfied_ratio_global)
                self.current_step += 1
                self.current_time += self.step_duration
                gini_index.append(
                    EscooterRDCEnv.calculate_gini_coefficient(
                        vehicle_counts=self.global_vehicle_counts.copy()
                    )
                )

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
