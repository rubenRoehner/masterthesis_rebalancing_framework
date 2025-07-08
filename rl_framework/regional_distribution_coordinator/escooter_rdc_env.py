"""
escooter_rdc_env.py

E-scooter Regional Distribution Coordinator Environment.
This gymnasium environment simulates e-scooter fleet management across multiple communities,
supporting demand forecasting, vehicle rebalancing, and reward calculation based on service quality.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import torch

from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider import DemandProvider


class EscooterRDCEnv(gym.Env):
    """Gymnasium environment for e-scooter regional distribution coordination.

    This environment simulates the management of an e-scooter fleet across multiple zones and communities,
    with demand patterns, vehicle rebalancing actions, and multi-objective reward calculation.
    """

    def __init__(
        self,
        num_communities: int,
        n_zones: int,
        action_values: List[int],
        max_steps: int,
        fleet_size: int,
        zone_community_map: pd.DataFrame,
        zone_index_map: dict[str, int],
        community_index_map: dict[str, int],
        pickup_demand_forecaster: DemandForecaster,
        dropoff_demand_forecaster: DemandForecaster,
        pickup_demand_provider: DemandProvider,
        dropoff_demand_provider: DemandProvider,
        device: torch.device,
        start_time: datetime = datetime(2025, 2, 11, 14, 0),
        step_duration: timedelta = timedelta(minutes=60),
        reward_weight_demand: float = 1.0,
        reward_weight_rebalancing: float = -0.5,
        reward_weight_gini: float = -0.1,
    ) -> None:
        """Initialize the E-scooter RDC environment.

        Args:
            num_communities: number of communities in the environment
            n_zones: total number of zones across all communities
            action_values: list of possible rebalancing action values
            max_steps: maximum number of steps per episode
            fleet_size: total number of vehicles in the fleet
            zone_community_map: DataFrame mapping zones to communities
            zone_index_map: mapping from zone IDs to indices
            community_index_map: mapping from community IDs to indices
            pickup_demand_forecaster: forecaster for pickup demand
            dropoff_demand_forecaster: forecaster for dropoff demand
            pickup_demand_provider: provider for actual pickup demand
            dropoff_demand_provider: provider for actual dropoff demand
            device: PyTorch device for tensor operations
            start_time: simulation start time
            step_duration: duration of each simulation step
            reward_weight_demand: weight for demand satisfaction in reward
            reward_weight_rebalancing: weight for rebalancing cost in reward
            reward_weight_gini: weight for distribution equality in reward

        Returns:
            None

        Raises:
            None
        """
        super(EscooterRDCEnv, self).__init__()

        self.device = device

        self.num_communities = num_communities
        self.n_zones = n_zones
        self.zone_community_map = zone_community_map
        self.zone_index_map = zone_index_map

        self.community_index_map = community_index_map

        self.reward_weight_demand = reward_weight_demand
        self.reward_weight_rebalancing = reward_weight_rebalancing
        self.reward_weight_gini = reward_weight_gini

        self.pickup_demand_forecaster: DemandForecaster = pickup_demand_forecaster
        self.dropoff_demand_forecaster: DemandForecaster = dropoff_demand_forecaster

        self.pickup_demand_provider: DemandProvider = pickup_demand_provider
        self.dropoff_demand_provider: DemandProvider = dropoff_demand_provider

        self.current_pickup_demand_forecast = np.zeros(num_communities)
        self.current_dropoff_demand_forecast = np.zeros(num_communities)

        self.current_vehicle_counts = np.zeros(n_zones, dtype=int)
        self.fleet_size = fleet_size

        self.action_values = action_values

        self.max_steps = max_steps
        self.current_step = 0

        self.start_time = start_time
        self.step_duration = step_duration

    def get_vehicle_counts_per_community(self) -> np.ndarray:
        """Get the total number of vehicles in each community.

        Args:
            None

        Returns:
            np.ndarray: array of vehicle counts per community

        Raises:
            None
        """
        vehicle_counts_per_community = np.zeros(self.num_communities, dtype=int)
        for i in range(self.n_zones):
            community_id = self.zone_community_map.iloc[i]["community_index"]
            community_index = self.community_index_map[community_id]
            vehicle_counts_per_community[
                community_index
            ] += self.current_vehicle_counts[i]
        return vehicle_counts_per_community

    def get_state_observation(self) -> torch.Tensor:
        """Get the current state observation for the agent.

        The observation includes demand forecasts and vehicle counts for each community.

        Args:
            None

        Returns:
            torch.Tensor: current state observation

        Raises:
            None
        """
        observation = []
        vehicle_counts_per_community = self.get_vehicle_counts_per_community()

        for i in range(self.num_communities):
            community_observation = [
                self.current_pickup_demand_forecast[i],
                self.current_dropoff_demand_forecast[i],
                vehicle_counts_per_community[i],
            ]
            observation.extend(community_observation)

        return torch.tensor(observation, dtype=torch.float32, device=self.device)

    def calculate_current_time(self) -> datetime:
        """Calculate the current simulation time based on step count.

        Args:
            None

        Returns:
            datetime: current simulation time

        Raises:
            None
        """
        return self.start_time + self.step_duration * self.current_step

    def generate_demand_forecast(self, current_time: datetime) -> None:
        """Generate demand forecasts for the current time.

        Args:
            current_time: current simulation time

        Returns:
            None

        Raises:
            None
        """
        time_of_day = current_time.hour
        day = current_time.day
        month = current_time.month

        self.current_pickup_demand_forecast = (
            self.pickup_demand_forecaster.predict_demand_per_community(
                time_of_day=time_of_day, day=day, month=month
            )
        )
        self.current_dropoff_demand_forecast = (
            self.dropoff_demand_forecaster.predict_demand_per_community(
                time_of_day=time_of_day, day=day, month=month
            )
        )

    def calculate_reward(
        self,
        satisfied_demand_ratio,
        total_vehicles_rebalanced,
        gini_coefficient,
    ) -> float:
        """Calculate the multi-objective reward for the current step.

        Args:
            satisfied_demand_ratio: ratio of satisfied pickup demand
            total_vehicles_rebalanced: total number of vehicles moved
            gini_coefficient: Gini coefficient of vehicle distribution

        Returns:
            float: calculated reward value between 0 and 1

        Raises:
            None
        """
        gini_r = 1.0 - gini_coefficient

        reb_ratio = total_vehicles_rebalanced / self.fleet_size
        reb_r = 1.0 - (reb_ratio**2)

        wd = self.reward_weight_demand
        wr = self.reward_weight_rebalancing
        wg = self.reward_weight_gini
        total_w = wd + wr + wg

        reward = (wd * satisfied_demand_ratio + wr * reb_r + wg * gini_r) / total_w
        return float(np.clip(reward, 0.0, 1.0))

    def calculate_gini_coefficient(self) -> float:
        """Calculate the Gini coefficient of vehicle distribution across zones.

        Args:
            None

        Returns:
            float: Gini coefficient (0 = perfect equality, 1 = maximum inequality)

        Raises:
            None
        """
        array = np.array(self.current_vehicle_counts, dtype=float)
        size = array.size
        mean = array.mean()

        if mean == 0:
            return 0

        diff = np.abs(np.subtract.outer(array, array))
        gini = diff.sum() / (2 * size**2 * mean)
        return gini

    @staticmethod
    def handle_rebalancing(
        actions: list[int],
        current_vehicle_counts: np.ndarray,
        community_index_map: dict[str, int],
        zone_community_map: pd.DataFrame,
        zone_index_map: dict[str, int],
    ) -> tuple[int, np.ndarray]:
        """Handle vehicle rebalancing actions across communities.

        Args:
            actions: list of rebalancing actions for each community
            current_vehicle_counts: current vehicle counts per zone
            community_index_map: mapping from community IDs to indices
            zone_community_map: DataFrame mapping zones to communities
            zone_index_map: mapping from zone IDs to indices

        Returns:
            tuple: (total_vehicles_rebalanced, updated_vehicle_counts)

        Raises:
            None
        """
        temp_vehicle_counts = current_vehicle_counts.copy()

        add_requests = {i: val for i, val in enumerate(actions) if val > 0}
        remove_requests = {i: val for i, val in enumerate(actions) if val < 0}

        total_add_requested = sum(add_requests.values())
        total_remove_requested = -sum(remove_requests.values())

        if total_add_requested == 0 or total_remove_requested == 0:
            return 0, temp_vehicle_counts

        if total_add_requested > total_remove_requested:
            scale_factor = total_remove_requested / total_add_requested
            for i in add_requests:
                add_requests[i] = int(add_requests[i] * scale_factor)
        elif total_remove_requested > total_add_requested:
            scale_factor = total_add_requested / total_remove_requested
            for i in remove_requests:
                remove_requests[i] = int(remove_requests[i] * scale_factor)

        total_to_add = sum(add_requests.values())
        total_to_remove = -sum(remove_requests.values())
        vehicles_to_move = min(total_to_add, total_to_remove)

        if vehicles_to_move == 0:
            return 0, temp_vehicle_counts

        vehicle_pool = 0
        vehicles_removed_count = 0

        for community_index, allocation in remove_requests.items():
            community_id = [
                cid
                for cid, cidx in community_index_map.items()
                if cidx == community_index
            ][0]
            zones_in_community = (
                zone_community_map[
                    zone_community_map["community_index"] == community_id
                ]["grid_index"]
                .map(zone_index_map)
                .tolist()
            )

            vehicles_to_remove_from_comm = min(
                -allocation, vehicles_to_move - vehicles_removed_count
            )

            for _ in range(vehicles_to_remove_from_comm):
                fullest_zones = sorted(
                    zones_in_community,
                    key=lambda z: temp_vehicle_counts[z],
                    reverse=True,
                )
                for zone_idx in fullest_zones:
                    if temp_vehicle_counts[zone_idx] > 0:
                        temp_vehicle_counts[zone_idx] -= 1
                        vehicle_pool += 1
                        vehicles_removed_count += 1
                        break

        total_vehicles_rebalanced = vehicle_pool

        vehicles_added_count = 0
        for community_index, allocation in add_requests.items():
            community_id = [
                cid
                for cid, cidx in community_index_map.items()
                if cidx == community_index
            ][0]
            zones_in_community = (
                zone_community_map[
                    zone_community_map["community_index"] == community_id
                ]["grid_index"]
                .map(zone_index_map)
                .tolist()
            )

            vehicles_to_add_to_comm = min(
                allocation, vehicle_pool - vehicles_added_count
            )

            for _ in range(vehicles_to_add_to_comm):
                if vehicle_pool > 0:
                    emptiest_zone = min(
                        zones_in_community, key=lambda z: temp_vehicle_counts[z]
                    )
                    temp_vehicle_counts[emptiest_zone] += 1
                    vehicle_pool -= 1
                    vehicles_added_count += 1

        return total_vehicles_rebalanced, temp_vehicle_counts

    def simulate_demand(self, current_time: datetime) -> float:
        """Simulate pickup and dropoff demand for the current time step.

        Args:
            current_time: current simulation time

        Returns:
            float: ratio of satisfied pickup demand

        Raises:
            None
        """
        total_satisfied_demand = 0
        pickup_demand = self.pickup_demand_provider.get_demand_per_zone(
            time_of_day=current_time.hour,
            day=current_time.day,
            month=current_time.month,
        )
        dropoff_demand = self.dropoff_demand_provider.get_demand_per_zone(
            time_of_day=current_time.hour,
            day=current_time.day,
            month=current_time.month,
        )

        satisfied_pickups = np.zeros(self.n_zones, dtype=int)
        for i in range(self.n_zones):
            satisfied_pickups[i] = min(pickup_demand[i], self.current_vehicle_counts[i])
            total_satisfied_demand += satisfied_pickups[i]
            self.current_vehicle_counts[i] -= satisfied_pickups[i]

        total_satisfied_pickups = total_satisfied_demand
        if total_satisfied_pickups > 0 and dropoff_demand.sum() > 0:
            dropoff_proportions = dropoff_demand / dropoff_demand.sum()
            satisfied_dropoffs = (dropoff_proportions * total_satisfied_pickups).astype(
                int
            )

            remaining = total_satisfied_pickups - satisfied_dropoffs.sum()
            if remaining > 0:
                fractional_parts = (
                    dropoff_proportions * total_satisfied_pickups
                ) - satisfied_dropoffs
                top_zones = np.argsort(fractional_parts)[-remaining:]
                satisfied_dropoffs[top_zones] += 1

            self.current_vehicle_counts += satisfied_dropoffs

        offered_demand = sum(pickup_demand)
        satisfied_demand_ratio = (
            total_satisfied_demand / offered_demand if offered_demand > 0 else 1.0
        )
        return satisfied_demand_ratio

    def step(self, action: List[int]) -> tuple[torch.Tensor, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: list of action indices for each community

        Returns:
            tuple: (observation, reward, terminated, truncated, info)

        Raises:
            AssertionError: if negative vehicle counts are detected
        """
        actual_action_allocations = [self.action_values[a] for a in action]

        total_vehicles_rebalanced, self.current_vehicle_counts = (
            EscooterRDCEnv.handle_rebalancing(
                actions=actual_action_allocations,
                current_vehicle_counts=self.current_vehicle_counts,
                community_index_map=self.community_index_map,
                zone_community_map=self.zone_community_map,
                zone_index_map=self.zone_index_map,
            )
        )

        satisfied_demand_ratio = self.simulate_demand(
            current_time=self.calculate_current_time()
        )

        reward = self.calculate_reward(
            satisfied_demand_ratio=satisfied_demand_ratio,
            total_vehicles_rebalanced=total_vehicles_rebalanced,
            gini_coefficient=self.calculate_gini_coefficient(),
        )

        self.current_step += 1
        self.generate_demand_forecast(current_time=self.calculate_current_time())

        next_observation = self.get_state_observation()
        terminated = False
        truncated = self.current_step >= self.max_steps

        assert np.all(
            self.current_vehicle_counts >= 0
        ), "Negative scooter count detected!"

        info = {
            "total_vehicles_rebalanced": total_vehicles_rebalanced,
        }

        return next_observation, reward, terminated, truncated, info

    def initialize_vehicle_counts(self) -> None:
        """Initialize vehicle counts evenly across all zones.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        self.current_vehicle_counts = np.full(
            self.n_zones,
            self.fleet_size // self.n_zones,
            dtype=int,
        )
        remaining_vehicles = self.fleet_size % self.n_zones
        for i in range(remaining_vehicles):
            self.current_vehicle_counts[i] += 1

    def reset(self, *, seed=None, options=None) -> tuple[torch.Tensor, dict]:
        """Reset the environment to a new episode.

        Args:
            seed: random seed for the episode
            options: additional reset options

        Returns:
            tuple: (initial_observation, info)

        Raises:
            None
        """
        super().reset(seed=seed, options=options)

        self.start_time = self.pickup_demand_provider.get_random_start_time(
            max_steps=self.max_steps, step_duration=self.step_duration
        )
        self.current_step = 0
        self.initialize_vehicle_counts()
        self.generate_demand_forecast(current_time=self.start_time)

        info = {}

        return self.get_state_observation(), info

    def render(self) -> None:
        """Render the current environment state.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        print("Rendering the environment state.")
        print(f"Step: {self.current_step}")

    def close(self) -> None:
        """Close the environment.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        print("Closing the environment.")
