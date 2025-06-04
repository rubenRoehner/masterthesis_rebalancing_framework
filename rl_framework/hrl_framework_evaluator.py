import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import List, Dict
from demand_forecasting.demand_forecaster import DemandForecaster
from demand_provider.demand_provider import DemandProvider
from regional_distribution_coordinator.regional_distribution_coordinator import (
    RegionalDistributionCoordinator,
)
from regional_distribution_coordinator.escooter_rdc_env import EscooterRDCEnv
from user_incentive_coordinator.user_incentive_coordinator import (
    UserIncentiveCoordinator,
)
from user_incentive_coordinator.escooter_uic_env import EscooterUICEnv


class HRLFrameworkEvaluator:
    def __init__(
        self,
        rdc_agent: RegionalDistributionCoordinator,
        rdc_env: EscooterRDCEnv,
        uic_agents: list[UserIncentiveCoordinator],
        uic_envs: list[EscooterUICEnv],
        zone_community_map: pd.DataFrame,
        community_index_map: dict[str, int],
        zone_index_map: dict[str, int],
        pickup_forecaster: DemandForecaster,
        dropoff_forecaster: DemandForecaster,
        pickup_provider: DemandProvider,
        dropoff_provider: DemandProvider,
        fleet_size: int,
        start_time: datetime,
        max_steps: int,
        step_duration: timedelta,
        user_willingness: list[float],
        zone_neighbor_map: dict[str, list[str]],
    ):
        self.rdc_agent = rdc_agent
        self.rdc_env = rdc_env
        self.uic_agents = uic_agents
        self.uic_envs = uic_envs
        self.zone_community_map = zone_community_map
        self.community_index_map = community_index_map
        self.zone_index_map = zone_index_map
        self.pickup_forecaster = pickup_forecaster
        self.dropoff_forecaster = dropoff_forecaster
        self.pickup_provider = pickup_provider
        self.dropoff_provider = dropoff_provider
        self.fleet_size = fleet_size
        self.start_time = start_time
        self.max_steps = max_steps
        self.user_willingness = user_willingness

        self.zone_neighbor_map = zone_neighbor_map
        self.n_communities = len(community_index_map)
        self.n_total_zones = zone_community_map.shape[0]

        self.zones_in_community: List[List[int]] = [
            [] for _ in range(self.n_communities)
        ]
        for idx, row in zone_community_map.iterrows():
            community_id = row["community_index"]
            community_index = community_index_map[community_id]
            zone_index_global = zone_index_map[row["grid_index"]]
            self.zones_in_community[community_index].append(zone_index_global)

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

    def get_vehicle_counts_per_community(self) -> np.ndarray:
        vehicle_counts_per_community = np.zeros(self.n_communities, dtype=int)
        for i in range(self.n_communities):
            vehicle_counts_per_community[i] = self.global_vehicle_counts[
                self.zones_in_community[i]
            ].sum()
        return vehicle_counts_per_community

    def get_rdc_state_observation(self) -> torch.Tensor:
        observation = []
        current_pickup_demand_forecast = (
            self.pickup_forecaster.predict_demand_per_community(
                time_of_day=self.current_time.hour,
                day=self.current_time.day,
                month=self.current_time.month,
            )
        )
        current_dropoff_demand_forecast = (
            self.dropoff_forecaster.predict_demand_per_community(
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

        return torch.tensor(
            observation, dtype=torch.float32, device=self.rdc_agent.device
        )

    def get_uic_state_observation(self, community_index: int) -> Dict[str, np.ndarray]:
        zones = self.zones_in_community[community_index]
        vehicle_counts = self.global_vehicle_counts[zones].astype(int).copy()

        pickup_forecast_full = self.pickup_forecaster.predict_demand_per_zone(
            time_of_day=self.current_time.hour,
            day=self.current_time.day,
            month=self.current_time.month,
        )
        dropoff_forecast_full = self.dropoff_forecaster.predict_demand_per_zone(
            time_of_day=self.current_time.hour,
            day=self.current_time.day,
            month=self.current_time.month,
        )

        pickup_demand_forecast = pickup_forecast_full[zones].astype(np.float32)
        dropoff_demand_forecast = dropoff_forecast_full[zones].astype(np.float32)

        return {
            "pickup_demand": pickup_demand_forecast,
            "dropoff_demand": dropoff_demand_forecast,
            "current_vehicle_counts": vehicle_counts,
        }

    def evaluate(
        self,
    ) -> Dict:
        satisfied_ratio = []
        while self.current_step < self.max_steps:
            # --- RDC operator based rebalancing ---
            obs_rdc = self.get_rdc_state_observation()

            with torch.no_grad():
                rdc_action_inidces = self.rdc_agent.select_action(obs_rdc)

            rdc_action_inidces = [int(a) for a in rdc_action_inidces]

            rdc_allocations = [
                self.rdc_agent.action_values[a] for a in rdc_action_inidces
            ]

            _, self.current_vehicle_counts = EscooterRDCEnv.handle_rebalancing(
                actions=rdc_allocations,
                current_vehicle_counts=self.current_vehicle_counts,
                community_index_map=self.community_index_map,
                zone_community_map=self.zone_community_map,
                zone_index_map=self.zone_index_map,
            )

            satisfied_per_community = [0 for _ in range(self.n_communities)]
            offered_per_community = [0 for _ in range(self.n_communities)]

            for community_index in range(self.n_communities):
                # --- UIC user-based rebalancing ---
                community_id = self.uic_envs[community_index].community_id

                uic_observation = self.get_uic_state_observation(
                    community_index=community_index
                )

                uic_action, _ = self.uic_agents[community_index].model.predict(
                    uic_observation, deterministic=True
                )

                local_dropoff_demand = (
                    self.dropoff_provider.get_demand_per_zone_community(
                        time_of_day=self.current_time.hour,
                        day=self.current_time.day,
                        month=self.current_time.month,
                        community_id=community_id,
                    )
                )
                offered_demand = len(local_dropoff_demand)
                local_pickup_demand = (
                    self.pickup_provider.get_demand_per_zone_community(
                        time_of_day=self.current_time.hour,
                        day=self.current_time.day,
                        month=self.current_time.month,
                        community_id=community_id,
                    )
                )

                zones = self.zones_in_community[community_index]

                zone_index_map: dict[str, int] = {}
                for i, row in self.zone_community_map[
                    self.zone_community_map["community_index"] == community_id
                ].iterrows():
                    zone_index_map.update({row["grid_index"]: i})

                zone_ids = list(self.zone_index_map.keys())
                zone_id_to_local = {
                    zone_id: idx for idx, zone_id in enumerate(zone_ids)
                }
                local_vehicle_counts = (
                    self.global_vehicle_counts[zones].astype(int).copy()
                )

                local_dropoff_demand, _ = EscooterUICEnv.handle_incentives(
                    action=uic_action.tolist(),
                    dropoff_demand=local_dropoff_demand,
                    zone_ids=zone_ids,
                    zone_id_to_local=zone_id_to_local,
                    zone_neighbor_map=self.zone_neighbor_map,
                    user_willingness=self.user_willingness,
                )

                local_vehicle_counts, total_satisfied_demand = (
                    EscooterUICEnv.update_vehicle_counts(
                        n_zones=len(local_vehicle_counts),
                        pickup_demand=local_pickup_demand,
                        dropoff_demand=local_dropoff_demand,
                        current_vehicle_counts=local_vehicle_counts,
                    )
                )

                satisfied_per_community[community_index] = total_satisfied_demand
                offered_per_community[community_index] = offered_demand

            total_satisfied_all = sum(satisfied_per_community)
            total_offered_all = sum(offered_per_community)
            if total_offered_all > 0:
                satisfied_ratio_global = total_satisfied_all / float(total_offered_all)
            else:
                satisfied_ratio_global = 1.0

            satisfied_ratio.append(satisfied_ratio_global)
            self.current_step += 1
            self.current_time += self.step_duration

        return {
            "mean_satisfied_ratio": np.mean(satisfied_ratio),
        }
