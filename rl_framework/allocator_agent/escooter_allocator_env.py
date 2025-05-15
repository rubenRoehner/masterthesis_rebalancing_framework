import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
from typing import List
import torch

from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider import DemandProvider


class EscooterAllocatorEnv(gym.Env):
    def __init__(
        self,
        num_communities: int,
        features_per_community: int,
        action_values: List[int],
        max_steps: int,
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
        super(EscooterAllocatorEnv, self).__init__()

        self.device = device

        self.num_communities = num_communities
        self.features_per_community = features_per_community
        self.allocator_state_dim = num_communities * features_per_community

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
        self.current_vehicle_counts = np.zeros(num_communities)

        self.action_values = action_values
        self.action_space = spaces.MultiDiscrete([len(action_values)] * num_communities)

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.allocator_state_dim,), dtype=np.float32
        )

        self.max_steps = max_steps
        self.current_step = 0

        self.start_time = start_time
        self.step_duration = step_duration

    def get_state_observation(self):
        observation = []

        for i in range(self.num_communities):
            community_observation = [
                self.current_pickup_demand_forecast[i],
                self.current_dropoff_demand_forecast[i],
                self.current_vehicle_counts[i],
            ]
            observation.extend(community_observation)

        return torch.tensor(observation, dtype=torch.float32, device=self.device)

    def calculate_current_time(self):
        # Calculate the current time based on start_time, step_duration, and current_step
        return self.start_time + self.step_duration * self.current_step

    def generate_demand_forecast(self, current_time: datetime):
        # Get the current time of day, day of week, and month based on start_time, tep_duration and current_step
        time_of_day = current_time.hour
        day = current_time.day
        month = current_time.month

        self.current_pickup_demand_forecast = (
            self.pickup_demand_forecaster.predict_demand_per_community(
                time_of_day=time_of_day, day=day, month=month
            )
        )
        self.current_dropoff_demand_forecast = (
            self.pickup_demand_forecaster.predict_demand_per_community(
                time_of_day=time_of_day, day=day, month=month
            )
        )

    def calculate_reward(
        self, total_satisfied_demand, rebalancing_cost, gini_coefficient
    ):
        return (
            self.reward_weight_demand * total_satisfied_demand
            + self.reward_weight_rebalancing * rebalancing_cost
            + self.reward_weight_gini * gini_coefficient
        )

    def calculate_gini_coefficient(self):
        # based on self.current_vehicle_counts
        return 0.0  # Placeholder for Gini coefficient calculation

    def step(self, action: List[int]):
        # --- map action to vehicle allocation ---
        actual_action_allocations = [self.action_values[a] for a in action]

        # --- update vehicle counts based on action ---
        total_vehicles_rebalanced = 0
        temp_vehicle_counts = self.current_vehicle_counts.copy()
        for i in range(self.num_communities):
            allocation = actual_action_allocations[i]

            temp_vehicle_counts[i] += allocation

            # --- verify vehicle counts are valid e.g. > 0 ---
            if temp_vehicle_counts[i] < 0:
                # If the allocation results in negative vehicle counts, set it to zero
                # and adjust the allocation accordingly for proper reward calculation
                allocation = -temp_vehicle_counts[i]
                temp_vehicle_counts[i] = 0

            total_vehicles_rebalanced += abs(allocation)

        self.current_vehicle_counts = temp_vehicle_counts

        # --- simulate demand based on historical data ---
        total_satisfied_demand = 0
        pickup_demand = self.pickup_demand_provider.get_demand_per_community(
            time_of_day=self.start_time.hour,
            day=self.start_time.day,
            month=self.start_time.month,
        )
        dropoff_demand = self.dropoff_demand_provider.get_demand_per_community(
            time_of_day=self.start_time.hour,
            day=self.start_time.day,
            month=self.start_time.month,
        )

        # Calculate total satisfied demand based on vehicle counts and demand
        for i in range(self.num_communities):
            updated_vehicle_count = self.current_vehicle_counts[i] + dropoff_demand[i]
            if updated_vehicle_count > pickup_demand[i]:
                updated_vehicle_count -= pickup_demand[i]
                total_satisfied_demand += pickup_demand[i]
            else:
                total_satisfied_demand += updated_vehicle_count
                updated_vehicle_count = 0
            self.current_vehicle_counts[i] = updated_vehicle_count

        # --- calculate reward ---
        rebalancing_cost = total_vehicles_rebalanced * self.operator_rebalancing_cost

        reward = self.calculate_reward(
            total_satisfied_demand=total_satisfied_demand,
            rebalancing_cost=rebalancing_cost,
            gini_coefficient=self.calculate_gini_coefficient(),
        )

        # --- Prepare next state ---
        self.current_step += 1
        current_time = self.calculate_current_time()
        self.generate_demand_forecast(current_time=current_time)

        next_observation = self.get_state_observation()
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {}

        return next_observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_step = 0
        self.current_vehicle_counts = np.zeros(self.num_communities)
        self.generate_demand_forecast(current_time=self.start_time)

        info = {}

        return self.get_state_observation(), info

    def render(self):
        print("Rendering the environment state.")
        print(f"Step: {self.current_step}")

    def close(self):
        print("Closing the environment.")
