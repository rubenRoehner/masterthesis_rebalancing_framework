# rl_framework/user_incentive_coordinator/escooter_uic_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Callable, List, Dict
import torch

from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider import DemandProvider
from regional_distribution_coordinator.regional_distribution_coordinator import (
    RegionalDistributionCoordinator,
)
from regional_distribution_coordinator.escooter_rdc_env import EscooterRDCEnv


class EscooterUICEnv(gym.Env):
    """
    Gymnasium environment for e-scooter user incentive coordination.

    This environment simulates the entire service area (all communities) to provide a
    stable and realistic training ground. It trains a single UIC agent for its
    assigned community while accounting for the global vehicle state.

    An optional pre-trained RDC agent can be passed to simulate its impact on the
    global fleet distribution during training.
    """

    def __init__(
        self,
        community_id: str,
        fleet_size: int,
        zone_community_map: pd.DataFrame,
        community_index_map: Dict[str, int],
        zone_index_map: Dict[str, int],
        zone_neighbor_map: Dict[str, List[str]],
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
        rdc_agent: RegionalDistributionCoordinator,
    ) -> None:
        """
        Initializes the UIC environment that simulates the entire service area.
        """
        super().__init__()
        self.active_community_id = community_id
        self.fleet_size = fleet_size
        self.zone_community_map = zone_community_map
        self.community_index_map = community_index_map
        self.zone_index_map = zone_index_map
        self.zone_neighbor_map = zone_neighbor_map
        self.user_willingness_fn = user_willingness_fn
        self.pickup_demand_forecaster = pickup_demand_forecaster
        self.dropoff_demand_forecaster = dropoff_demand_forecaster
        self.pickup_demand_provider = pickup_demand_provider
        self.dropoff_demand_provider = dropoff_demand_provider
        self.max_steps = max_steps
        self.start_time = start_time
        self.step_duration = step_duration
        self.reward_weight_demand = reward_weight_demand
        self.reward_weight_rebalancing = reward_weight_rebalancing
        self.reward_weight_gini = reward_weight_gini
        self.device = device

        self.rdc_agent = rdc_agent

        self.n_total_zones = len(self.zone_index_map)
        self.n_communities = len(self.community_index_map)

        self._populate_helper_maps()

        self.n_zones_local = len(self.local_zone_ids)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_zones_local,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "pickup_demand": spaces.Box(
                    low=0, high=np.inf, shape=(self.n_zones_local,), dtype=np.float32
                ),
                "dropoff_demand": spaces.Box(
                    low=0, high=np.inf, shape=(self.n_zones_local,), dtype=np.float32
                ),
                "current_vehicle_counts": spaces.Box(
                    low=0,
                    high=self.fleet_size,
                    shape=(self.n_zones_local,),
                    dtype=np.int32,
                ),
            }
        )

        self.global_vehicle_counts = np.zeros(self.n_total_zones, dtype=int)
        self.full_time_index_available: pd.Index[datetime] = (
            self.pickup_demand_provider.demand_data.index
        )
        self.demand_trace_length = len(self.full_time_index_available)
        self.current_step = 0
        self.current_time = self.start_time
        self.start_offset = 0

    def _populate_helper_maps(self):
        """Populates helper maps for the active community and for the global RDC simulation."""
        community_zones_df = self.zone_community_map[
            self.zone_community_map["community_index"] == self.active_community_id
        ]
        self.local_zone_ids = list(community_zones_df["grid_index"])
        self.local_zone_id_to_local_idx = {
            zone_id: i for i, zone_id in enumerate(self.local_zone_ids)
        }
        self.local_zone_global_indices = [
            self.zone_index_map[zid] for zid in self.local_zone_ids
        ]

        self.zones_in_community_global: Dict[str, List[int]] = {
            cid: [
                self.zone_index_map[zid]
                for zid in self.zone_community_map[
                    self.zone_community_map["community_index"] == cid
                ]["grid_index"]
            ]
            for cid in self.community_index_map.keys()
        }

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Gets the observation for the currently active community using FORECASTERS."""
        self.current_time = self._calculate_current_time()
        pickup_forecast_full = self.pickup_demand_forecaster.predict_demand_per_zone(
            self.current_time.hour, self.current_time.day, self.current_time.month
        )
        dropoff_forecast_full = self.dropoff_demand_forecaster.predict_demand_per_zone(
            self.current_time.hour, self.current_time.day, self.current_time.month
        )

        local_vehicle_counts = self.global_vehicle_counts[
            self.local_zone_global_indices
        ]
        local_pickup_forecast = pickup_forecast_full[self.local_zone_global_indices]
        local_dropoff_forecast = dropoff_forecast_full[self.local_zone_global_indices]

        return {
            "pickup_demand": local_pickup_forecast.astype(np.float32),
            "dropoff_demand": local_dropoff_forecast.astype(np.float32),
            "current_vehicle_counts": local_vehicle_counts.astype(np.int32),
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Executes one full time step, including RDC action."""
        if self.rdc_agent:
            self._simulate_rdc_step()

        self.current_time = self._calculate_current_time()

        pickup_demand = self.pickup_demand_provider.get_demand_per_zone_community(
            self.current_time.hour,
            self.current_time.day,
            self.current_time.month,
            self.active_community_id,
        )
        dropoff_demand = self.dropoff_demand_provider.get_demand_per_zone_community(
            self.current_time.hour,
            self.current_time.day,
            self.current_time.month,
            self.active_community_id,
        )

        local_vehicle_counts = self.global_vehicle_counts[
            self.local_zone_global_indices
        ].copy()

        local_zone_neighbor_map = {
            zid: [
                n
                for n in self.zone_neighbor_map.get(zid, [])
                if n in self.local_zone_id_to_local_idx
            ]
            for zid in self.local_zone_ids
        }

        dropoff_demand, vehicles_rebalanced = self.handle_incentives(
            action=action,
            dropoff_demand=dropoff_demand,
            zone_ids=self.local_zone_ids,
            zone_id_to_local=self.local_zone_id_to_local_idx,
            zone_neighbor_map=local_zone_neighbor_map,
            user_willingness_fn=self.user_willingness_fn,
        )
        new_local_counts, satisfied_demand, offered_demand = self.update_vehicle_counts(
            n_zones=self.n_zones_local,
            pickup_demand=pickup_demand,
            dropoff_demand=dropoff_demand,
            current_vehicle_counts=local_vehicle_counts,
        )

        self.global_vehicle_counts[self.local_zone_global_indices] = new_local_counts

        gini_coefficient = EscooterRDCEnv.calculate_gini_coefficient(new_local_counts.copy())
        reward = self.calculate_reward(
            satisfied_demand, offered_demand, vehicles_rebalanced, gini_coefficient
        )

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        next_observation = self.get_observation()

        assert (
            self.global_vehicle_counts.sum() == self.fleet_size
        ), f"Vehicle counts do not match the total fleet size after demand simulation. Expected: {self.fleet_size}, Actual: {self.global_vehicle_counts.sum()}"

        return (
            next_observation,
            reward,
            terminated,
            False,
            {"vehicles_rebalanced": vehicles_rebalanced},
        )

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        """Resets the entire simulation state for a new episode."""
        super().reset(seed=seed)
        self.current_step = 0
        self.start_offset = self.np_random.integers(0, self.demand_trace_length)
        self.current_time = self._calculate_current_time()

        self.global_vehicle_counts = np.full(
            self.n_total_zones, self.fleet_size // self.n_total_zones, dtype=int
        )
        remainder = self.fleet_size % self.n_total_zones
        if remainder > 0:
            idxs = self.np_random.choice(self.n_total_zones, size=remainder, replace=False)
            self.global_vehicle_counts[idxs] += 1

        return self.get_observation(), {}

    def _simulate_rdc_step(self):
        """Prepares RDC observation and updates global vehicle counts using the static method."""
        obs_rdc = self._get_rdc_observation()
        with torch.no_grad():
            rdc_action_indices = self.rdc_agent.select_action(obs_rdc)
        rdc_allocations = [
            self.rdc_agent.action_values[int(a)] for a in rdc_action_indices
        ]

        _, self.global_vehicle_counts = EscooterRDCEnv.handle_rebalancing(
            actions=rdc_allocations,
            current_vehicle_counts=self.global_vehicle_counts,
            community_index_map=self.community_index_map,
            zone_community_map=self.zone_community_map,
            zone_index_map=self.zone_index_map,
        )

    def _get_rdc_observation(self) -> torch.Tensor:
        """Constructs the RDC observation using FORECASTERS and the current global state."""
        vehicle_counts_per_community = np.zeros(self.n_communities, dtype=int)
        for cid, cidx in self.community_index_map.items():
            vehicle_counts_per_community[cidx] = self.global_vehicle_counts[
                self.zones_in_community_global[cid]
            ].sum()

        pickup_forecast = self.pickup_demand_forecaster.predict_demand_per_community(
            self.current_time.hour, self.current_time.day, self.current_time.month
        )
        dropoff_forecast = self.dropoff_demand_forecaster.predict_demand_per_community(
            self.current_time.hour, self.current_time.day, self.current_time.month
        )

        observation = []
        for i in range(self.n_communities):
            observation.extend(
                [
                    pickup_forecast[i],
                    dropoff_forecast[i],
                    vehicle_counts_per_community[i],
                ]
            )

        return torch.tensor(observation, dtype=torch.float32, device=self.device)

    def _calculate_current_time(self) -> datetime:
        """Calculates the current simulation time based on step count and random offset."""
        idx = (self.start_offset + self.current_step) % self.demand_trace_length
        return self.full_time_index_available[idx]

    def calculate_reward(
        self,
        total_satisfied_demand: int,
        offered_demand: int,
        total_vehicles_rebalanced: int,
        gini_coefficient: float,
    ) -> float:
        """Calculates the multi-objective reward for the current step."""
        norm_satisified_demand = (
            total_satisfied_demand / offered_demand if offered_demand > 0 else 1.0
        )
        gini_reward = 1.0 - gini_coefficient

        local_fleet_size = int(self.global_vehicle_counts[self.local_zone_global_indices].sum())

        rebalancing_ratio = (
            total_vehicles_rebalanced / local_fleet_size if local_fleet_size > 0 else 0
        )
        rebalancing_reward = 1.0 - rebalancing_ratio

        total_weight = (
            self.reward_weight_demand
            + self.reward_weight_rebalancing
            + self.reward_weight_gini
        )
        if total_weight == 0:
            return 0.0

        reward = (
            norm_satisified_demand * self.reward_weight_demand
            + rebalancing_reward * self.reward_weight_rebalancing
            + gini_reward * self.reward_weight_gini
        ) / total_weight

        return float(np.clip(reward, 0, 1))

    @staticmethod
    def calculate_gini_coefficient(vehicle_counts: np.ndarray) -> float:
        """Calculates the Gini coefficient for a given array of vehicle counts."""
        array = np.array(vehicle_counts, dtype=float)
        if np.amin(array) < 0:
            array -= np.amin(array)
        if array.sum() == 0:
            return 0

        array += 0.0000001

        mad = np.abs(np.subtract.outer(array, array)).mean()
        rmad = mad / np.mean(array)
        return 0.5 * rmad

    @staticmethod
    def handle_incentives(
        action: np.ndarray,
        dropoff_demand: np.ndarray,
        zone_ids: List[str],
        zone_id_to_local: dict[str, int],
        zone_neighbor_map: dict[str, List[str]],
        user_willingness_fn: Callable[[float], float],
    ) -> tuple[np.ndarray, int]:
        """Handles user incentives to influence dropoff behavior."""
        total_vehicles_rebalanced = 0
        modified_dropoff_demand = dropoff_demand.copy()
        initial_dropoff_sum = modified_dropoff_demand.sum()

        for local_idx, zone_id in enumerate(zone_ids):
            raw_neighbors = zone_neighbor_map.get(zone_id, [])
            neighbor_local_idxs = [
                zone_id_to_local[n] for n in raw_neighbors if n in zone_id_to_local
            ]

            if not neighbor_local_idxs:
                continue

            self_incentive = action[local_idx]
            candidate_local_idxs = [local_idx] + neighbor_local_idxs
            candidate_incentives = action[candidate_local_idxs]

            best_rel_pos = int(np.argmax(candidate_incentives))
            best_local_index = candidate_local_idxs[best_rel_pos]
            best_incentive = float(candidate_incentives[best_rel_pos])

            delta = best_incentive - float(self_incentive)
            if best_local_index != local_idx and delta > 0 and modified_dropoff_demand[local_idx] > 0:
                p_move = float(user_willingness_fn(delta))
                n = int(modified_dropoff_demand[local_idx])

                n_rebalanced = int(n * p_move)
                modified_dropoff_demand[local_idx] -= n_rebalanced
                modified_dropoff_demand[best_local_index] += n_rebalanced
                total_vehicles_rebalanced += n_rebalanced

        assert (
            modified_dropoff_demand.sum() == initial_dropoff_sum
        ), "Dropoff demand does not match the initial sum after incentive handling."

        return modified_dropoff_demand, total_vehicles_rebalanced

    @staticmethod
    def update_vehicle_counts(
        n_zones: int,
        pickup_demand: np.ndarray,
        dropoff_demand: np.ndarray,
        current_vehicle_counts: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        """Updates vehicles counts based on pickup and dropoff demand."""
        pickup_demand = pickup_demand.astype(int, copy=False)
        dropoff_demand = dropoff_demand.astype(int, copy=False)
        current_vehicle_counts = current_vehicle_counts.astype(int, copy=False)

        fleet_size = int(current_vehicle_counts.sum())

        satisfied_pickups = np.minimum(pickup_demand, current_vehicle_counts)
        total_satisfied_pickups = int(satisfied_pickups.sum())

        new_vehicle_counts = current_vehicle_counts - satisfied_pickups

        if total_satisfied_pickups > 0:
            drop_sum = float(dropoff_demand.sum())

            if drop_sum > 0.0:
                target = (dropoff_demand / drop_sum) * total_satisfied_pickups  # float
                satisfied_dropoffs = np.floor(target).astype(int)
                remainder = total_satisfied_pickups - int(satisfied_dropoffs.sum())
                if remainder > 0:
                    residual = target - satisfied_dropoffs
                    top_idx = np.argsort(residual)[-remainder:]
                    satisfied_dropoffs[top_idx] += 1
            else:
                satisfied_dropoffs = satisfied_pickups.copy()

            new_vehicle_counts += satisfied_dropoffs

        assert int(new_vehicle_counts.sum()) == fleet_size, \
            "Vehicle counts do not match total fleet size after update."

        total_satisfied_demand = total_satisfied_pickups
        total_offered_demand = int(pickup_demand.sum())

        return new_vehicle_counts, total_satisfied_demand, total_offered_demand

    def render(self):
        """Renders the current environment state."""
        print(
            f"Step: {self.current_step}, Time: {self.current_time}, Community: {self.active_community_id}"
        )
        print(f"Global Vehicle Count: {self.global_vehicle_counts.sum()}")

    def close(self):
        """Closes the environment."""
        pass
