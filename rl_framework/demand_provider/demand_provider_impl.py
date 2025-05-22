from demand_provider.demand_provider import DemandProvider
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DemandProviderImpl(DemandProvider):
    """
    Pickup demand provider module.
    """

    def __init__(
        self,
        num_communities: int,
        num_zones: int,
        zone_community_map: pd.DataFrame,
        demand_data_path: str,
    ):
        super().__init__(num_communities, num_zones, zone_community_map)
        self.demand_data_path: str = demand_data_path
        self.demand_data: pd.DataFrame = pd.read_pickle(demand_data_path)

        # Ensure demand_data index is DatetimeIndex
        self.demand_data.index = pd.to_datetime(self.demand_data.index)
        self.demand_data.sort_index(inplace=True)
        # Reindex to full hourly range, filling missing hours with zeros
        full_idx = pd.date_range(
            start=self.demand_data.index.min(),
            end=self.demand_data.index.max(),
            freq="h",
        )
        self.demand_data = self.demand_data.reindex(full_idx, fill_value=0)

    def get_random_start_time(
        self, max_steps: int, step_duration: timedelta
    ) -> datetime:
        """
        Returns a random datetime from the demand_data index such that
        (start_time + step_duration * max_steps) <= last timestamp.
        """
        idxs = self.demand_data.index
        max_delta = step_duration * max_steps
        valid = idxs[idxs + max_delta <= idxs[-1]]
        return valid[np.random.randint(len(valid))]

    def get_demand_per_zone(self, time_of_day: int, day: int, month: int) -> np.ndarray:
        """
        Get demand per zone for a given time of day, day of week, and month.
        """
        # Convert to datetime
        dt = datetime(year=2025, month=month, day=day, hour=time_of_day)

        # Filter demand data for the specified time
        demand_data_filtered = self.demand_data.loc[dt]

        # Return demand per zone as a numpy array
        return demand_data_filtered.to_numpy()

    def get_demand_per_community(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        """
        Get demand per community for a given time of day, day of week, and month.
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

        # group by community index and sum the demand; drop grid_index
        community_demand = zone_demand.groupby("community_index").sum()
        community_demand.sort_index(inplace=True)
        community_demand.drop(columns=["grid_index"], inplace=True)

        # return demand per community as a numpy array
        return community_demand["demand"].to_numpy()

    def get_demand_per_zone_community(
        self, time_of_day: int, day: int, month: int, community_id: str
    ) -> np.ndarray:
        """
        Get demand per zone for a given time of day, day of week, and month.
        """
        # Convert to datetime
        dt = datetime(year=2025, month=month, day=day, hour=time_of_day)

        # Filter demand data for the specified time
        demand_data_filtered = self.demand_data.loc[dt]

        # Filter demand data for the specified community
        community_mask = self.zone_community_map["community_index"] == community_id
        zone_ids = self.zone_community_map.loc[community_mask, "grid_index"].tolist()

        community_demand = demand_data_filtered.loc[zone_ids]

        community_demand = community_demand.reindex(sorted(zone_ids))

        return community_demand.to_numpy()
