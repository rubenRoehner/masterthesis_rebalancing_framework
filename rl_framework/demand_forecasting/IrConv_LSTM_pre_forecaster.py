"""
IrConv_LSTM_pre_forecaster.py

Historical data-based demand forecaster implementation.
This module provides demand forecasting by retrieving historical demand patterns
from stored data, serving as a baseline or preprocessing step for more complex forecasting models.
"""

from demand_forecasting.demand_forecaster import DemandForecaster
import numpy as np
import pandas as pd


class IrConvLstmDemandPreForecaster(DemandForecaster):
    """Historical data-based demand forecaster.

    This forecaster retrieves historical demand patterns from stored data
    for the specified time periods, providing baseline forecasts or serving
    as input preprocessing for more sophisticated forecasting models.
    """

    def __init__(
        self,
        num_communities: int,
        num_zones: int,
        zone_community_map: pd.DataFrame,
        demand_data_path: str,
    ) -> None:
        """Initialize the IrConvLstmDemandPreForecaster with historical data.

        Args:
            num_communities: total number of communities in the system
            num_zones: total number of zones across all communities
            zone_community_map: DataFrame mapping zones to their communities
            demand_data_path: path to the pickled historical demand data file

        Returns:
            None

        Raises:
            None
        """
        super().__init__(num_communities, num_zones, zone_community_map)

        self.demand_data_path = demand_data_path
        self.demand_data = pd.read_pickle(demand_data_path)

        self.demand_data.index = pd.to_datetime(self.demand_data.index)
        self.demand_data.sort_index(inplace=True)
        full_idx = pd.date_range(
            start=self.demand_data.index.min(),
            end=self.demand_data.index.max(),
            freq="h",
        )
        self.demand_data = self.demand_data.reindex(full_idx, fill_value=0)

    def _get_target_index(self, time_of_day: int, day: int, month: int) -> int:
        """Get the index of the target timestamp in the historical data.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            int: index position of the matching timestamp

        Raises:
            ValueError: if no matching historical data is found
        """
        matching_timestamps = self.demand_data.index[
            (self.demand_data.index.hour == time_of_day)
            & (self.demand_data.index.day == day)
            & (self.demand_data.index.month == month)
        ]

        if matching_timestamps.empty:
            raise ValueError(
                f"No historical data found for time_of_day={time_of_day}, "
                f"day={day}, month={month} in the provided demand data."
            )

        current_dt = matching_timestamps[-1]

        target_idx_array = self.demand_data.index.get_indexer([current_dt])

        return target_idx_array[0]

    def predict_demand_per_zone(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        """Predict demand for all zones using historical data patterns.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: historical demand values for each zone

        Raises:
            ValueError: if no matching historical data is found
        """

        return self.demand_data.iloc[
            self._get_target_index(time_of_day, day, month)
        ].to_numpy()

    def predict_demand_per_community(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        """Predict aggregated demand for all communities using historical data.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: historical aggregated demand values for each community

        Raises:
            ValueError: if no matching historical data is found
        """
        zone_demand_values = self.predict_demand_per_zone(time_of_day, day, month)
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

        # group by community index and sum the demand
        community_demand = zone_demand.groupby("community_index").sum()
        community_demand.sort_index(inplace=True)
        community_demand.drop(columns=["grid_index"], inplace=True)
        # return demand per community as a numpy array
        return community_demand["demand"].to_numpy()

    def predict_demand_per_zone_community(
        self, time_of_day: int, day: int, month: int, community_id: str
    ) -> np.ndarray:
        """Predict demand for zones within a specific community using historical data.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year
            community_id: ID of the target community

        Returns:
            np.ndarray: historical demand values for zones in the specified community

        Raises:
            ValueError: if no matching historical data is found
        """
        zone_demand_values = self.predict_demand_per_zone(time_of_day, day, month)
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

        # filter by community id
        community_demand = zone_demand[zone_demand["community_index"] == community_id][
            "demand"
        ].to_numpy()
        return community_demand
