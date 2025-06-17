"""
demand_provider_impl.py

Implementation of the DemandProvider interface.
This module provides concrete implementation for accessing historical e-scooter demand data
from pickled DataFrame files, supporting zone-level and community-level demand queries.
"""

from demand_provider.demand_provider import DemandProvider
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DemandProviderImpl(DemandProvider):
    """Concrete implementation of DemandProvider using historical data files.

    This provider loads demand data from pickled DataFrames and provides
    temporal demand patterns for zones and communities based on historical records.
    """

    def __init__(
        self,
        num_communities: int,
        num_zones: int,
        zone_community_map: pd.DataFrame,
        demand_data_path: str,
    ) -> None:
        """Initialize the DemandProviderImpl with historical data.

        Args:
            num_communities: total number of communities in the system
            num_zones: total number of zones across all communities
            zone_community_map: DataFrame mapping zones to their communities
            demand_data_path: path to the pickled demand data file

        Returns:
            None

        Raises:
            None
        """
        super().__init__(num_communities, num_zones, zone_community_map)
        self.demand_data_path: str = demand_data_path
        self.demand_data: pd.DataFrame = pd.read_pickle(demand_data_path)

        self.demand_data.index = pd.to_datetime(self.demand_data.index)
        self.demand_data.sort_index(inplace=True)

        full_idx = pd.date_range(
            start=self.demand_data.index.min(),
            end=self.demand_data.index.max(),
            freq="h",
        )
        self.demand_data = self.demand_data.reindex(full_idx, fill_value=0)

    def get_random_start_time(
        self, max_steps: int, step_duration: timedelta
    ) -> datetime:
        """Get a random valid start time ensuring full episode can be completed.

        Args:
            max_steps: maximum number of steps in the episode
            step_duration: duration of each simulation step

        Returns:
            datetime: randomly selected start time from available data

        Raises:
            None
        """
        idxs = self.demand_data.index
        max_delta = step_duration * max_steps
        valid = idxs[idxs + max_delta <= idxs[-1]]
        return valid[np.random.randint(len(valid))]

    def get_demand_per_zone(self, time_of_day: int, day: int, month: int) -> np.ndarray:
        """Get historical demand values for all zones at specified time.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: demand values for each zone from historical data

        Raises:
            None
        """
        dt = datetime(year=2025, month=month, day=day, hour=time_of_day)

        demand_data_filtered = self.demand_data.loc[dt]

        return demand_data_filtered.to_numpy()

    def get_demand_per_community(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        """Get aggregated historical demand for all communities at specified time.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: aggregated demand values for each community

        Raises:
            None
        """
        zone_demand_values = self.get_demand_per_zone(time_of_day, day, month)
        zone_indices = sorted(self.zone_community_map["grid_index"].values)
        zone_demand = pd.DataFrame(
            zone_demand_values, index=zone_indices, columns=["demand"]
        )
        zone_demand = pd.merge(
            zone_demand,
            self.zone_community_map,
            left_index=True,
            right_on="grid_index",
            how="left",
        )

        community_demand = zone_demand.groupby("community_index").sum()
        community_demand.sort_index(inplace=True)
        community_demand.drop(columns=["grid_index"], inplace=True)

        return community_demand["demand"].to_numpy()

    def get_demand_per_zone_community(
        self, time_of_day: int, day: int, month: int, community_id: str
    ) -> np.ndarray:
        """Get historical demand for zones within a specific community.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year
            community_id: ID of the target community

        Returns:
            np.ndarray: demand values for zones in the specified community

        Raises:
            None
        """
        dt = datetime(year=2025, month=month, day=day, hour=time_of_day)

        demand_data_filtered = self.demand_data.loc[dt]

        community_mask = self.zone_community_map["community_index"] == community_id
        zone_ids = self.zone_community_map.loc[community_mask, "grid_index"].tolist()

        community_demand = demand_data_filtered.loc[zone_ids]

        community_demand = community_demand.reindex(sorted(zone_ids))

        return community_demand.to_numpy()
