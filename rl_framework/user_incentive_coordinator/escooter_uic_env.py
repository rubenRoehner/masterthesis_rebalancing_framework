import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
from typing import List
import torch

from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider import DemandProvider


class EscooterUICEnv(gym.Env):
    def __init__(
        self,
        community_id: str,
        n_zones: int,
        fleet_size: int,
        zone_neighbor_map: dict[int, List[int]],  # int -> List[int]
        zone_index_map: dict[str, int],  # str -> int
        user_willingness: list[float],
        pickup_demand_forecaster: DemandForecaster,
        dropoff_demand_forecaster: DemandForecaster,
        pickup_demand_provider: DemandProvider,
        dropoff_demand_provider: DemandProvider,
        max_incentive: float,
        incentive_levels: int,
        max_steps: int,
        start_time: datetime,
        step_duration: timedelta,
        reward_weight_demand: float,
        reward_weight_rebalancing: float,
        reward_weight_gini: float,
        device: torch.device,
    ):
        """
        Args:
            community_id (str): The ID of the community.
            n_zones (int): The number of zones in the community.
            fleet_size (int): The size of the fleet.
            pickup_demand_forecaster (DemandForecaster): The demand forecaster for pickup.
            dropoff_demand_forecaster (DemandForecaster): The demand forecaster for dropoff.
            pickup_demand_provider (DemandProvider): The demand provider for pickup.
            dropoff_demand_provider (DemandProvider): The demand provider for dropoff.
            max_incentive (float): The maximum incentive that can be offered.
            incentive_levels (int): The number of incentive levels.
            max_steps (int): The maximum number of steps in an episode.
            start_time (datetime): The start time of the simulation.
            step_duration (timedelta): The duration of each step in the simulation.
            reward_weight_demand (float): The weight for the demand satisfaction in the reward.
            reward_weight_rebalancing (float): The weight for the rebalancing costs in the reward.
            reward_weight_gini (float): The weight for the Gini coefficient in the reward.
        """
        super(EscooterUICEnv, self).__init__()
        self.device = device

        self.community_id = community_id
        self.n_zones = n_zones
        self.fleet_size = fleet_size
        self.zone_neighbor_map = zone_neighbor_map
        self.zone_index_map = zone_index_map
        self.user_willingness = user_willingness

        self.pickup_demand_forecaster = pickup_demand_forecaster
        self.dropoff_demand_forecaster = dropoff_demand_forecaster
        self.pickup_demand_provider = pickup_demand_provider
        self.dropoff_demand_provider = dropoff_demand_provider

        self.max_incentive = max_incentive
        self.incentive_levels = incentive_levels

        self.current_pickup_demand_forecast = np.zeros(n_zones, dtype=np.float32)
        self.current_dropoff_demand_forecast = np.zeros(n_zones, dtype=np.float32)

        self.current_vehicle_counts = np.zeros(n_zones, dtype=int)

        # Actions: For each zone, assign an incentive level (0 to incentive_levels-1)
        self.action_space = spaces.MultiDiscrete([incentive_levels] * self.n_zones)

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

    def get_state_observation(self):
        return {
            "pickup_demand": self.current_pickup_demand_forecast,
            "dropoff_demand": self.current_dropoff_demand_forecast,
            "current_vehicle_counts": self.current_vehicle_counts,
        }

    def calculate_current_time(self):
        # Calculate the current time based on start_time, step_duration, and current_step
        return self.start_time + self.step_duration * self.current_step

    def generate_demand_forecast(self, current_time: datetime):
        # Get the current time of day, day of week, and month based on start_time, step_duration and current_step
        time_of_day = current_time.hour
        day = current_time.day
        month = current_time.month

        # Generate demand forecasts for pickup and dropoff
        self.pickup_demand_forecast = (
            self.pickup_demand_forecaster.predict_demand_per_zone_community(
                time_of_day, day, month, self.community_id
            )
        )
        self.dropoff_demand_forecast = (
            self.dropoff_demand_forecaster.predict_demand_per_zone_community(
                time_of_day, day, month, self.community_id
            )
        )

    def calculate_reward(
        self,
        total_satisfied_demand: int,
        offered_demand: int,
        total_vehicles_rebalanced: int,
        gini_coefficient: float,
    ) -> float:
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
        array = np.array(self.current_vehicle_counts, dtype=float)
        size = array.size
        mean = array.mean()

        if mean == 0:
            return 0

        diff = np.abs(np.subtract.outer(array, array))
        gini = diff.sum() / (2 * size**2 * mean)
        return gini

    def handle_incentives(
        self, action: List[int], dropoff_demand: np.ndarray
    ) -> tuple[np.ndarray, int]:
        # Update vehicle counts based on the action taken
        total_vehicles_rebalanced = 0
        initial_dropoff_demand = dropoff_demand.copy()
        for i in range(self.n_zones):
            neighbor_zones = self.zone_neighbor_map.get(i, [])
            # Get incentive level for the neighbor zones
            nieghbor_zones_incentive = np.ndarray(
                shape=(len(neighbor_zones),),
                dtype=np.float32,
            )
            for j, neighbor in enumerate(neighbor_zones):
                nieghbor_zones_incentive[j] = action[neighbor]

            # zone with highest incentive
            max_incentive_index = np.argmax(nieghbor_zones_incentive)
            max_incentive = nieghbor_zones_incentive[max_incentive_index]

            if max_incentive > 0:
                # Calculate the number of vehicles to be rebalanced
                vehicles_to_rebalance = int(
                    dropoff_demand[i] * max_incentive / self.max_incentive
                )
                # Update the dropoff for the current zone and the neighbor zone
                dropoff_demand[i] -= vehicles_to_rebalance
                dropoff_demand[max_incentive_index] += vehicles_to_rebalance

                total_vehicles_rebalanced += vehicles_to_rebalance

        # Ensure that droppoff demand sum did not change
        if np.sum(dropoff_demand) != np.sum(initial_dropoff_demand):
            raise ValueError(
                "Dropoff demand sum changed after applying incentives. "
                f"Initial sum: {np.sum(initial_dropoff_demand)}, "
                f"Final sum: {np.sum(dropoff_demand)}"
            )
        return dropoff_demand, total_vehicles_rebalanced

    def step(self, action: List[int]):
        current_time = self.calculate_current_time()

        # --- simulate demand based on historical data ---
        total_satisfied_demand = 0
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

        # update dropoff demand based on the incentives
        dropoff_demand, total_vehicles_rebalanced = self.handle_incentives(
            action, dropoff_demand
        )

        # Calculate total satisfied demand based on vehicle counts and demand
        for i in range(self.n_zones):
            updated_vehicle_count = self.current_vehicle_counts[i] + dropoff_demand[i]
            if updated_vehicle_count > pickup_demand[i]:
                updated_vehicle_count -= pickup_demand[i]
                total_satisfied_demand += pickup_demand[i]
            else:
                total_satisfied_demand += updated_vehicle_count
                updated_vehicle_count = 0
            self.current_vehicle_counts[i] = updated_vehicle_count

        reward = self.calculate_reward(
            total_satisfied_demand=total_satisfied_demand,
            offered_demand=sum(pickup_demand),
            total_vehicles_rebalanced=total_vehicles_rebalanced,
            gini_coefficient=self.calculate_gini_coefficient(),
        )

        # --- Prepare next state ---
        self.current_step += 1
        self.generate_demand_forecast(current_time=current_time)

        next_observation = self.get_state_observation()
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "total_vehicles_rebalanced": total_vehicles_rebalanced,
        }

        return next_observation, reward, terminated, truncated, info

    def initialize_vehicle_counts(self):
        # Initialize vehicle counts based on the fleet size and number of communities
        # Distribute the fleet size evenly across the zones
        self.current_vehicle_counts = np.full(
            self.n_zones,
            self.fleet_size // self.n_zones,
            dtype=int,
        )
        # Distribute the remaining vehicles to the first few zones
        remaining_vehicles = self.fleet_size % self.n_zones
        for i in range(remaining_vehicles):
            self.current_vehicle_counts[i] += 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_step = 0
        self.initialize_vehicle_counts()
        self.generate_demand_forecast(current_time=self.start_time)

        info = {}

        return self.get_state_observation(), info

    def render(self):
        print("Rendering the environment state.")
        print(f"Step: {self.current_step}")

    def close(self):
        print("Closing the environment.")
