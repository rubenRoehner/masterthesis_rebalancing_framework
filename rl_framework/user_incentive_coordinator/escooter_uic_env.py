"""
escooter_uic_env.py

E-scooter User Incentive Coordinator Environment.
This gymnasium environment simulates user incentive-based e-scooter rebalancing within a single community,
where the agent learns to provide incentives that influence user dropoff behavior.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Callable, List
import torch

from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider import DemandProvider


class EscooterUICEnv(gym.Env):
    """Gymnasium environment for e-scooter user incentive coordination.

    This environment simulates incentive-based rebalancing within a single community,
    where the agent provides incentives to users to influence their dropoff locations,
    thereby achieving better fleet distribution without direct vehicle movement.
    """

    def __init__(
        self,
        community_id: str,
        n_zones: int,
        fleet_size: int,
        zone_neighbor_map: dict[str, List[str]],
        zone_index_map: dict[str, int],
        user_willingness_fn: Callable[[float], float],
        pickup_demand_forecaster: DemandForecaster,
        dropoff_demand_forecaster: DemandForecaster,
        pickup_demand_provider: DemandProvider,
        dropoff_demand_provider: DemandProvider,
        max_steps: int,
        start_time: datetime,
        step_duration: timedelta,
        reward_weight_demand: float,
        reward_weight_rebalancing: float,
        reward_weight_gini: float,
        device: torch.device,
    ) -> None:
        """Initialize the E-scooter User Incentive Coordinator environment.

        Args:
            community_id: ID of the community this environment manages
            n_zones: number of zones within the community
            fleet_size: total number of vehicles in the community fleet
            zone_neighbor_map: mapping from zone IDs to their neighbor zone IDs
            zone_index_map: mapping from zone IDs to their indices
            user_willingness_fn: function mapping incentive level to user compliance probability
            pickup_demand_forecaster: forecaster for pickup demand patterns
            dropoff_demand_forecaster: forecaster for dropoff demand patterns
            pickup_demand_provider: provider for actual pickup demand data
            dropoff_demand_provider: provider for actual dropoff demand data
            max_steps: maximum number of steps per episode
            start_time: simulation start time
            step_duration: duration of each simulation step
            reward_weight_demand: weight for demand satisfaction in reward calculation
            reward_weight_rebalancing: weight for rebalancing efficiency in reward calculation
            reward_weight_gini: weight for distribution equality in reward calculation
            device: PyTorch device for tensor operations

        Returns:
            None

        Raises:
            None
        """
        super(EscooterUICEnv, self).__init__()
        self.device = device

        self.community_id = community_id
        self.n_zones = n_zones
        self.fleet_size = fleet_size
        self.zone_neighbor_map = zone_neighbor_map
        self.zone_index_map = zone_index_map
        self.user_willingness_fn = user_willingness_fn

        self.zone_ids = list(self.zone_index_map.keys())

        self.zone_id_to_local = {
            zone_id: idx for idx, zone_id in enumerate(self.zone_ids)
        }

        self.pickup_demand_forecaster = pickup_demand_forecaster
        self.dropoff_demand_forecaster = dropoff_demand_forecaster
        self.pickup_demand_provider = pickup_demand_provider
        self.dropoff_demand_provider = dropoff_demand_provider

        self.current_pickup_demand_forecast = np.zeros(n_zones, dtype=np.float32)
        self.current_dropoff_demand_forecast = np.zeros(n_zones, dtype=np.float32)

        self.current_vehicle_counts = np.zeros(n_zones, dtype=int)

        self.action_space = spaces.Box(0.0, 1.0, shape=(n_zones,), dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "pickup_demand": spaces.Box(
                    low=0, high=np.inf, shape=(self.n_zones,), dtype=np.float32
                ),
                "dropoff_demand": spaces.Box(
                    low=0, high=np.inf, shape=(self.n_zones,), dtype=np.float32
                ),
                "current_vehicle_counts": spaces.Box(
                    low=0, high=self.fleet_size, shape=(self.n_zones,), dtype=np.int32
                ),
            }
        )

        self.reward_weight_demand = reward_weight_demand
        self.reward_weight_rebalancing = reward_weight_rebalancing
        self.reward_weight_gini = reward_weight_gini

        self.max_steps = max_steps
        self.current_step = 0

        self.start_time = start_time
        self.step_duration = step_duration
        self.start_offset = 0

        self.full_time_index_available: pd.Index[datetime] = (
            pickup_demand_provider.demand_data.index
        )
        self.demand_trace_length = len(self.full_time_index_available)

    def get_state_observation(self) -> dict:
        """Get the current state observation for the agent.

        Args:
            None

        Returns:
            dict: observation containing demand forecasts and vehicle counts

        Raises:
            None
        """
        return {
            "pickup_demand": self.current_pickup_demand_forecast,
            "dropoff_demand": self.current_dropoff_demand_forecast,
            "current_vehicle_counts": self.current_vehicle_counts,
        }

    def calculate_current_time(self) -> datetime:
        """Calculate the current simulation time based on step count and offset.

        Args:
            None

        Returns:
            datetime: current simulation time

        Raises:
            None
        """
        idx = (self.start_offset + self.current_step) % self.demand_trace_length
        return self.full_time_index_available[idx]

    def generate_demand_forecast(self, current_time: datetime) -> None:
        """Generate demand forecasts for the current time within the community.

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

        full_pickup_demand_forecast = (
            self.pickup_demand_forecaster.predict_demand_per_zone(
                time_of_day, day, month
            )
        )
        full_dropoff_demand_forecast = (
            self.dropoff_demand_forecaster.predict_demand_per_zone(
                time_of_day, day, month
            )
        )

        zone_inidices = list(self.zone_index_map.values())

        self.pickup_demand_forecast = full_pickup_demand_forecast[zone_inidices]
        self.dropoff_demand_forecast = full_dropoff_demand_forecast[zone_inidices]

    def calculate_reward(
        self,
        total_satisfied_demand: int,
        offered_demand: int,
        total_vehicles_rebalanced: int,
        gini_coefficient: float,
    ) -> float:
        """Calculate the multi-objective reward for the current step.

        Args:
            total_satisfied_demand: number of pickup requests satisfied
            offered_demand: total number of pickup requests
            total_vehicles_rebalanced: number of vehicles moved through incentives
            gini_coefficient: Gini coefficient of vehicle distribution

        Returns:
            float: calculated reward value between 0 and 1

        Raises:
            None
        """
        norm_satisified_demand = (
            total_satisfied_demand / offered_demand if offered_demand > 0 else 1
        )

        gini_reward = 1 - gini_coefficient

        rebalancing_ratio = total_vehicles_rebalanced / self.fleet_size
        rebalancing_reward = 1 - rebalancing_ratio

        total_weight = (
            self.reward_weight_demand
            + self.reward_weight_rebalancing
            + self.reward_weight_gini
        )

        reward = norm_satisified_demand * self.reward_weight_demand / total_weight
        reward += rebalancing_reward * self.reward_weight_rebalancing / total_weight
        reward += gini_reward * self.reward_weight_gini / total_weight

        return float(np.clip(reward, 0, 1))

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
    def handle_incentives(
        action: np.ndarray,
        dropoff_demand: np.ndarray,
        zone_ids: List[str],
        zone_id_to_local: dict[str, int],
        zone_neighbor_map: dict[str, List[str]],
        user_willingness_fn: Callable[[float], float],
    ) -> tuple[np.ndarray, int]:
        """Handle user incentives to influence dropoff behavior.

        Rebalances vehicles by shifting dropoff demand from each zone to the
        neighboring zone with the highest incentive, based on user willingness.

        Args:
            action: incentive levels for each zone (0-1 normalized)
            dropoff_demand: current dropoff demand per zone
            zone_ids: list of zone IDs in consistent order
            zone_id_to_local: mapping from zone IDs to local indices
            zone_neighbor_map: mapping from zone IDs to neighbor zone IDs
            user_willingness_fn: function mapping incentive to compliance probability

        Returns:
            tuple: (updated_dropoff_demand, total_vehicles_rebalanced)

        Raises:
            ValueError: if dropoff demand sum changes unexpectedly
        """
        total_vehicles_rebalanced = 0
        initial_dropoff = dropoff_demand.copy()

        for local_idx, zone_id in enumerate(zone_ids):
            raw_neighbors = zone_neighbor_map.get(zone_id, [])

            neighbor_local_idxs = [
                zone_id_to_local[n] for n in raw_neighbors if n in zone_id_to_local
            ]
            if not neighbor_local_idxs:
                continue

            incentives: list[float] = [action[idx] for idx in neighbor_local_idxs]

            best_position = np.argmax(incentives)
            best_local_index = neighbor_local_idxs[best_position]
            max_incentive = incentives[best_position]
            willingness = user_willingness_fn(max_incentive)

            if max_incentive > 0:
                n_scooter = int(dropoff_demand[local_idx] * willingness)
                dropoff_demand[local_idx] -= n_scooter
                dropoff_demand[best_local_index] += n_scooter
                total_vehicles_rebalanced += n_scooter

        if dropoff_demand.sum() != initial_dropoff.sum():
            raise ValueError(
                f"Dropoff demand sum changed: {initial_dropoff.sum()} → {dropoff_demand.sum()}"
            )

        return dropoff_demand, total_vehicles_rebalanced

    @staticmethod
    def update_vehicle_counts(
        n_zones: int,
        pickup_demand: np.ndarray,
        dropoff_demand: np.ndarray,
        current_vehicle_counts: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Update vehicle counts based on pickup and dropoff demand.

        Vehicle counts are preserved by only processing satisfied demand.
        Pickups can only be satisfied if vehicles are available in the zone.
        Dropoffs only occur for satisfied pickups and equal the total satisfied pickups.

        Args:
            n_zones: number of zones in the community
            pickup_demand: pickup demand per zone
            dropoff_demand: dropoff demand per zone (after incentive influence)
            current_vehicle_counts: current vehicle counts per zone

        Returns:
            tuple: (updated_vehicle_counts, total_satisfied_demand)

        Raises:
            None
        """
        total_satisfied_demand = 0
        satisfied_pickups = np.zeros(n_zones, dtype=int)

        for i in range(n_zones):
            satisfied_pickups[i] = min(pickup_demand[i], current_vehicle_counts[i])
            total_satisfied_demand += satisfied_pickups[i]

        updated_vehicle_counts = current_vehicle_counts - satisfied_pickups

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

            updated_vehicle_counts += satisfied_dropoffs

        return updated_vehicle_counts, total_satisfied_demand

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: incentive levels for each zone (0-1 normalized)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)

        Raises:
            None
        """
        current_time = self.calculate_current_time()

        pickup_demand = self.pickup_demand_provider.get_demand_per_zone_community(
            time_of_day=current_time.hour,
            day=current_time.day,
            month=current_time.month,
            community_id=self.community_id,
        )
        dropoff_demand = self.dropoff_demand_provider.get_demand_per_zone_community(
            time_of_day=current_time.hour,
            day=current_time.day,
            month=current_time.month,
            community_id=self.community_id,
        )

        dropoff_demand, total_vehicles_rebalanced = EscooterUICEnv.handle_incentives(
            action=action,
            dropoff_demand=dropoff_demand,
            zone_ids=self.zone_ids,
            zone_id_to_local=self.zone_id_to_local,
            zone_neighbor_map=self.zone_neighbor_map,
            user_willingness_fn=self.user_willingness_fn,
        )

        self.current_vehicle_counts, total_satisfied_demand = (
            EscooterUICEnv.update_vehicle_counts(
                n_zones=self.n_zones,
                pickup_demand=pickup_demand,
                dropoff_demand=dropoff_demand,
                current_vehicle_counts=self.current_vehicle_counts.copy(),
            )
        )

        reward = self.calculate_reward(
            total_satisfied_demand=total_satisfied_demand,
            offered_demand=sum(pickup_demand),
            total_vehicles_rebalanced=total_vehicles_rebalanced,
            gini_coefficient=self.calculate_gini_coefficient(),
        )

        self.current_step += 1
        self.generate_demand_forecast(current_time=self.calculate_current_time())

        next_observation = self.get_state_observation()
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "total_vehicles_rebalanced": total_vehicles_rebalanced,
        }

        return next_observation, reward, terminated, truncated, info

    def initialize_vehicle_counts(self) -> None:
        """Initialize vehicle counts evenly across all zones in the community.

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

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
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

        self.start_offset = self.np_random.integers(0, self.demand_trace_length)
        self.current_step = 0

        self.initialize_vehicle_counts()
        self.generate_demand_forecast(current_time=self.calculate_current_time())

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
