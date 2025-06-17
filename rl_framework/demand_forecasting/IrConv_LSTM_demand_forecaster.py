"""
IrConv_LSTM_demand_forecaster.py

IrConv-LSTM neural network-based demand forecaster implementation.
This module provides sophisticated demand forecasting using a pre-trained
Irregular Convolution LSTM model that captures spatial-temporal patterns in e-scooter demand.
"""

from demand_forecasting.demand_forecaster import DemandForecaster

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import sys

# import IrConv_LSTM model for torch.load
from demand_forecasting.IrConv_LSTM.model.irregular_convolution_LSTM import (
    Irregular_Convolution_LSTM,  # noqa: F401
    Extraction_spatial_features,  # noqa: F401
    Convolution_LSTM,  # noqa: F401
    irregular_convolution,  # noqa: F401
)


class IrConvLstmDemandForecaster(DemandForecaster):
    """Neural network-based demand forecaster using IrConv-LSTM architecture.

    This forecaster uses a pre-trained Irregular Convolution LSTM model to predict
    e-scooter demand patterns by analyzing closeness, period, and trend components
    of historical demand data.
    """

    def __init__(
        self,
        num_communities: int,
        num_zones: int,
        zone_community_map: pd.DataFrame,
        model_path: str,
        demand_data_path: str,
        closeness_size: int = 24,
        period_size: int = 7,
        trend_size: int = 2,
    ) -> None:
        """Initialize the IrConv-LSTM demand forecaster with pre-trained model.

        Args:
            num_communities: total number of communities in the system
            num_zones: total number of zones across all communities
            zone_community_map: DataFrame mapping zones to their communities
            model_path: path to the pre-trained IrConv-LSTM model file
            demand_data_path: path to the historical demand data file
            closeness_size: number of recent hours to consider (default: 24)
            period_size: number of recent days to consider (default: 7)
            trend_size: number of recent weeks to consider (default: 2)

        Returns:
            None

        Raises:
            None
        """
        super().__init__(num_communities, num_zones, zone_community_map)
        self.model_path = model_path
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        sys.path.append(
            "/home/ruroit00/rebalancing_framework/rl_framework/demand_forecasting/IrConv_LSTM"
        )

        self.model = torch.load(model_path, weights_only=False, map_location=device)
        self.model.eval()

        self.demand_data_path = demand_data_path
        self.demand_data = pd.read_pickle(
            demand_data_path
        )  # DataFrame: (time_steps, num_zones)

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
        # Scale data
        # Reshape demand_data.values to (num_samples, num_nodes, 1) for scaling
        data_to_scale = self.demand_data.values[:, :, np.newaxis]
        self.scaled_demand_data, self.data_max, self.data_min = self._maxminscaler_3d(
            data_to_scale
        )

        self.closeness_size = closeness_size
        self.period_size = period_size
        self.trend_size = trend_size

    def predict_demand_per_zone(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        """Predict demand for all zones using the IrConv-LSTM model.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: predicted demand values for each zone

        Raises:
            ValueError: if no matching historical data is found for context
        """

        prediction = self.perform_inference(
            self.get_closeness_data(time_of_day, day, month),
            self.get_period_data(time_of_day, day, month),
            self.get_trend_data(time_of_day, day, month),
        )

        prediction = (
            prediction * (self.data_max - self.data_min) + self.data_min
        ).reshape(self.num_zones, 1)

        return prediction

    def predict_demand_per_community(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        """Predict aggregated demand for all communities using the IrConv-LSTM model.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: predicted aggregated demand values for each community

        Raises:
            ValueError: if no matching historical data is found for context
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
        """Predict demand for zones within a specific community using the IrConv-LSTM model.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year
            community_id: ID of the target community

        Returns:
            np.ndarray: predicted demand values for zones in the specified community

        Raises:
            ValueError: if no matching historical data is found for context
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

    def _maxminscaler_3d(
        self, tensor_3d: np.ndarray, data_range=(0, 1)
    ) -> tuple[np.ndarray, float, float]:
        """Scale a 3D numpy array to a given range using global min/max normalization.

        Args:
            tensor_3d: 3D numpy array to be scaled
            data_range: target range for scaling (default: (0, 1))

        Returns:
            tuple: (scaled_array, original_max, original_min)

        Raises:
            None
        """
        scaler_max = np.max(tensor_3d)
        scaler_min = np.min(tensor_3d)

        if scaler_max == scaler_min:
            # All values in tensor_3d are the same. Scale to the start of the data_range.
            X_scaled = np.full_like(tensor_3d, data_range[0], dtype=np.float32)
        else:
            X_std = (tensor_3d - scaler_min) / (scaler_max - scaler_min)
            X_scaled = X_std * (data_range[1] - data_range[0]) + data_range[0]
        return X_scaled, scaler_max, scaler_min

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
        # Filter index based on time components
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

        current_dt = matching_timestamps[-1]  # Get the most recent matching timestamp

        # Get its integer position using get_indexer
        target_idx_array = self.demand_data.index.get_indexer([current_dt])

        return target_idx_array[0]

    def _get_historical_data_point(self, target_idx: int, offset: int) -> np.ndarray:
        """Retrieve a historical data point from scaled demand data.

        Args:
            target_idx: index of the target timestamp
            offset: number of time steps to go back from target

        Returns:
            np.ndarray: historical data point with shape (num_zones, 1)

        Raises:
            None
        """
        hist_idx = target_idx - offset
        if 0 <= hist_idx < len(self.scaled_demand_data):
            return self.scaled_demand_data[hist_idx]
        else:
            # Data not available for this historical point, return zeros
            return np.zeros((self.num_zones, 1), dtype=np.float32)

    def perform_inference(
        self,
        closeness_data_np: np.ndarray,
        period_data_np: np.ndarray,
        trend_data_np: np.ndarray,
    ) -> np.ndarray:
        """Perform inference using the IrConv-LSTM model.

        Args:
            closeness_data_np: closeness component data
            period_data_np: period component data
            trend_data_np: trend component data

        Returns:
            np.ndarray: model prediction output

        Raises:
            None
        """
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.rnn.LSTM):
                module.flatten_parameters()

        closeness_tensor = Variable(
            torch.FloatTensor(closeness_data_np).unsqueeze(0)
        ).to(device)
        period_tensor = Variable(torch.FloatTensor(period_data_np).unsqueeze(0)).to(
            device
        )
        trend_tensor = Variable(torch.FloatTensor(trend_data_np).unsqueeze(0)).to(
            device
        )

        with torch.no_grad():
            prediction_tensor = self.model(
                closeness_tensor, period_tensor, trend_tensor
            )

        return prediction_tensor.squeeze().cpu().numpy()

    def get_closeness_data(self, time_of_day: int, day: int, month: int) -> np.ndarray:
        """Extract closeness component data for the specified time.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: closeness data with shape (closeness_size, num_zones, 1)

        Raises:
            ValueError: if no matching historical data is found
        """
        target_idx = self._get_target_index(time_of_day, day, month)

        closeness_data_list = []
        for c_offset_val in range(
            self.closeness_size, 0, -1
        ):  # Iterates closeness_size, ..., 1
            # c_offset_val is the actual offset (e.g., 1 for T-1, 2 for T-2, ...)
            # The loop range(self.closeness_size, 0, -1) gives [C, C-1, ..., 1]
            # So data is [T-C, T-(C-1), ..., T-1]
            closeness_data_list.append(
                self._get_historical_data_point(target_idx, c_offset_val)
            )

        if (
            not closeness_data_list and self.closeness_size > 0
        ):  # Should only be empty if closeness_size is 0
            return np.zeros((self.closeness_size, self.num_zones, 1), dtype=np.float32)
        return np.array(closeness_data_list, dtype=np.float32)

    def get_period_data(self, time_of_day: int, day: int, month: int) -> np.ndarray:
        """Extract period component data for the specified time.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: period data with shape (period_size, num_zones, 1)

        Raises:
            ValueError: if no matching historical data is found
        """
        target_idx = self._get_target_index(time_of_day, day, month)

        period_data_list = []
        for p_val in range(self.period_size, 0, -1):  # Iterates period_size, ..., 1
            offset = p_val * 24  # 24 hours in a day
            period_data_list.append(self._get_historical_data_point(target_idx, offset))

        if not period_data_list and self.period_size > 0:
            return np.zeros((self.period_size, self.num_zones, 1), dtype=np.float32)
        return np.array(period_data_list, dtype=np.float32)

    def get_trend_data(self, time_of_day: int, day: int, month: int) -> np.ndarray:
        """Extract trend component data for the specified time.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: trend data with shape (trend_size, num_zones, 1)

        Raises:
            ValueError: if no matching historical data is found
        """
        target_idx = self._get_target_index(time_of_day, day, month)

        trend_data_list = []
        for t_val in range(self.trend_size, 0, -1):  # Iterates trend_size, ..., 1
            offset = t_val * 168  # 168 hours in a week (7 days)
            trend_data_list.append(self._get_historical_data_point(target_idx, offset))

        if not trend_data_list and self.trend_size > 0:
            return np.zeros((self.trend_size, self.num_zones, 1), dtype=np.float32)
        return np.array(trend_data_list, dtype=np.float32)
