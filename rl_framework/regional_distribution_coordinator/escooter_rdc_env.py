import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import torch

from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider import DemandProvider


class EscooterRDCEnv(gym.Env):
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
        operator_rebalancing_cost: float = 0.5,
        start_time: datetime = datetime(2025, 2, 11, 14, 0),
        step_duration: timedelta = timedelta(minutes=60),
        reward_weight_demand: float = 1.0,
        reward_weight_rebalancing: float = -0.5,
        reward_weight_gini: float = -0.1,
    ):
        super(EscooterRDCEnv, self).__init__()

        self.device = device

        self.num_communities = num_communities
        self.n_zones = n_zones
        self.zone_community_map = zone_community_map
        self.zone_index_map = zone_index_map

        self.community_index_map = community_index_map

        self.operator_rebalancing_cost = operator_rebalancing_cost

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
        vehicle_counts_per_community = np.zeros(self.num_communities, dtype=int)
        for i in range(self.n_zones):
            community_id = self.zone_community_map.iloc[i]["community_index"]
            community_index = self.community_index_map[community_id]
            vehicle_counts_per_community[
                community_index
            ] += self.current_vehicle_counts[i]
        return vehicle_counts_per_community

    def get_state_observation(self):
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

    def calculate_current_time(self):
        return self.start_time + self.step_duration * self.current_step

    def generate_demand_forecast(self, current_time: datetime):
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
    ):
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
        array = np.array(self.current_vehicle_counts, dtype=float)
        size = array.size
        mean = array.mean()

        if mean == 0:
            return 0

        diff = np.abs(np.subtract.outer(array, array))
        gini = diff.sum() / (2 * size**2 * mean)
        return gini

    def handle_rebalancing(self, actions: list[int]) -> int:
        total_vehicles_rebalanced = 0
        temp_vehicle_counts = self.current_vehicle_counts.copy()  # len self.n_zones
        for community_id, community_index in self.community_index_map.items():
            allocation = actions[community_index]

            zones_in_community = (
                self.zone_community_map[
                    self.zone_community_map["community_index"] == community_id
                ]["grid_index"]
                .map(self.zone_index_map)
                .tolist()
            )

            if allocation > 0:
                vehicles_to_be_added = allocation
                for _ in range(vehicles_to_be_added):
                    emptiest_zones = min(
                        zones_in_community, key=lambda z: temp_vehicle_counts[z]
                    )
                    temp_vehicle_counts[emptiest_zones] += 1
                    total_vehicles_rebalanced += 1

            elif allocation < 0:
                vehicles_to_be_removed = -allocation
                for _ in range(vehicles_to_be_removed):
                    fullest_zones = max(
                        zones_in_community, key=lambda z: temp_vehicle_counts[z]
                    )
                    if temp_vehicle_counts[fullest_zones] > 0:
                        temp_vehicle_counts[fullest_zones] -= 1
                        total_vehicles_rebalanced += 1

        self.current_vehicle_counts = temp_vehicle_counts
        return total_vehicles_rebalanced

    def simulate_demand(self, current_time: datetime) -> float:
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

        for i in range(self.n_zones):
            updated_vehicle_count = self.current_vehicle_counts[i] + dropoff_demand[i]
            if updated_vehicle_count > pickup_demand[i]:
                updated_vehicle_count -= pickup_demand[i]
                total_satisfied_demand += pickup_demand[i]
            else:
                total_satisfied_demand += updated_vehicle_count
                updated_vehicle_count = 0
            self.current_vehicle_counts[i] = updated_vehicle_count

        offered_demand = sum(pickup_demand)
        satisfied_demand_ratio = (
            total_satisfied_demand / offered_demand if offered_demand > 0 else 1.0
        )
        return satisfied_demand_ratio

    def step(self, action: List[int]):
        actual_action_allocations = [self.action_values[a] for a in action]

        total_vehicles_rebalanced = self.handle_rebalancing(actual_action_allocations)

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

    def initialize_vehicle_counts(self):
        self.current_vehicle_counts = np.full(
            self.n_zones,
            self.fleet_size // self.n_zones,
            dtype=int,
        )
        remaining_vehicles = self.fleet_size % self.n_zones
        for i in range(remaining_vehicles):
            self.current_vehicle_counts[i] += 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.start_time = self.pickup_demand_provider.get_random_start_time(
            max_steps=self.max_steps, step_duration=self.step_duration
        )
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
