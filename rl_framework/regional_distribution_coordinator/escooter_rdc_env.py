import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
from typing import List
import torch

from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider import DemandProvider


class EscooterRDCEnv(gym.Env):
    def __init__(
        self,
        num_communities: int,
        features_per_community: int,
        action_values: List[int],
        max_steps: int,
        fleet_size: int,
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

        self.current_vehicle_counts = np.zeros(num_communities, dtype=int)
        self.fleet_size = fleet_size

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
            self.dropoff_demand_forecaster.predict_demand_per_community(
                time_of_day=time_of_day, day=day, month=month
            )
        )

    def calculate_reward(
        self,
        total_satisfied_demand,
        offered_demand,
        total_vehicles_rebalanced,
        gini_coefficient,
    ):
        # 1) satisfaction ratio
        sat = total_satisfied_demand / offered_demand if offered_demand > 0 else 1.0

        # 2) fairness
        gini_r = 1.0 - gini_coefficient

        # 3) quadratic rebalancing cost + dispatch overhead
        reb_ratio = total_vehicles_rebalanced / self.fleet_size
        reb_r = 1.0 - (reb_ratio**2)

        # 4) convex combination
        wd = self.reward_weight_demand
        wr = self.reward_weight_rebalancing
        wg = self.reward_weight_gini
        total_w = wd + wr + wg

        reward = (wd * sat + wr * reb_r + wg * gini_r) / total_w
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

    def update_vehicle_counts(self, actions: np.ndarray) -> int:
        neg_rebalancing_operations = -actions[
            actions < 0
        ]  # positive array of how many to pull
        pos_rebalancing_operations = actions[
            actions > 0
        ]  # positive array of how many to add

        sum_neg_rebalancing_operations = neg_rebalancing_operations.sum()
        sum_pos_rebalancing_operations = pos_rebalancing_operations.sum()

        rebalancing_vol = min(
            sum_neg_rebalancing_operations, sum_pos_rebalancing_operations
        )

        # --- 2) proportionally scale & round to integers ---------
        if rebalancing_vol > 0:
            # scale rebalancing operations to the transfer volume
            neg_scaled = neg_rebalancing_operations * (
                rebalancing_vol / sum_neg_rebalancing_operations
            )
            pos_scaled = pos_rebalancing_operations * (
                rebalancing_vol / sum_pos_rebalancing_operations
            )

            # floor each and keep track of remainders
            neg_floor = np.floor(neg_scaled).astype(int)
            pos_floor = np.floor(pos_scaled).astype(int)
            rem_neg = int(rebalancing_vol - neg_floor.sum())
            rem_pos = int(rebalancing_vol - pos_floor.sum())

            # fractional parts to decide where to add the “leftover” vehicles
            neg_frac = neg_scaled - neg_floor
            pos_frac = pos_scaled - pos_floor

            neg_indices = np.where(actions < 0)[0]
            pos_indices = np.where(actions > 0)[0]

            # sort by descending fractional part
            neg_order = np.argsort(-neg_frac)
            pos_order = np.argsort(-pos_frac)

            # distribute the remainders to the highest fractions
            for j in range(rem_neg):
                neg_floor[neg_order[j]] += 1
            for j in range(rem_pos):
                pos_floor[pos_order[j]] += 1

            # rebuild an array of actual integer allocations
            actual_alloc = np.zeros_like(actions)
            actual_alloc[neg_indices] = -neg_floor
            actual_alloc[pos_indices] = pos_floor
        else:
            # nothing to move if either side is zero
            actual_alloc = np.zeros_like(actions)

        self.current_vehicle_counts += actual_alloc
        total_vehicles_rebalanced = int(np.abs(actual_alloc).sum())
        return total_vehicles_rebalanced

    def step(self, action: List[int]):
        current_time = self.calculate_current_time()

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
            time_of_day=current_time.hour,
            day=current_time.day,
            month=current_time.month,
        )
        dropoff_demand = self.dropoff_demand_provider.get_demand_per_community(
            time_of_day=current_time.hour,
            day=current_time.day,
            month=current_time.month,
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
        self.current_vehicle_counts = np.full(
            self.num_communities,
            self.fleet_size // self.num_communities,
            dtype=int,
        )
        remaining_vehicles = self.fleet_size % self.num_communities
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
