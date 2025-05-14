from demand_forecaster import DemandForecaster

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable


class IrConvLstmDemandForecaster(DemandForecaster):
    """
    Demand forecasting module using IrConv-LSTM model.
    This class is designed to predict demand for a given time of day, day of week, and month.
    It uses a pre-trained IrConv-LSTM model to perform the forecasting.

    Parameters
    ----------
    num_communities : int
        Number of communities.
    num_zones : int
        Number of zones.
    zone_community_map : np.ndarray
        A mapping of zones to communities. Shape: (num_zones, num_communities).
    model_path : str
        Path to the pre-trained IrConv-LSTM model.
    demand_data_path : str
        Path to the demand data. The data should be a pandas DataFrame with a DatetimeIndex.
    closeness_size : int, optional
        Size of the closeness data in hours. Default is 24. Should be the same as for training.
    period_size : int, optional
        Size of the period data in days. Default is 7. Should be the same as for training.
    trend_size : int, optional
        Size of the trend data in weeks. Default is 2. Should be the same as for training.
    """

    def __init__(
        self,
        num_communities: int,
        num_zones: int,
        zone_community_map: np.ndarray,
        model_path: str,
        demand_data_path: str,
        closeness_size: int = 24,
        period_size: int = 7,
        trend_size: int = 2,
    ):
        super().__init__(num_communities, num_zones, zone_community_map)
        self.model_path = model_path
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        self.model = torch.load(model_path, weights_only=False, map_location=device)
        self.model.eval()

        self.demand_data_path = demand_data_path
        self.demand_data = pd.read_pickle(
            demand_data_path
        )  # DataFrame: (time_steps, num_zones)

        # Ensure demand_data index is DatetimeIndex
        self.demand_data.index = pd.to_datetime(self.demand_data.index)
        # Scale data
        # Reshape demand_data.values to (num_samples, num_nodes, 1) for scaling
        data_to_scale = self.demand_data.values[:, :, np.newaxis]
        self.scaled_demand_data, self.data_max, self.data_min = self._maxminscaler_3d(
            data_to_scale
        )

        self.closeness_size = closeness_size
        self.period_size = period_size
        self.trend_size = trend_size

    def get_demand_per_zone(
        self, time_of_day: int, day_of_week: int, month: int
    ) -> np.ndarray:
        """
        Get demand per zone for the given time of day, day of week, and month.
        """

        prediction = self.perform_inference(
            self.get_closeness_data(time_of_day, day_of_week, month),
            self.get_period_data(time_of_day, day_of_week, month),
            self.get_trend_data(time_of_day, day_of_week, month),
        )
        # Rescale the prediction
        prediction = (
            prediction * (self.data_max - self.data_min) + self.data_min
        ).reshape(self.num_zones, 1)

        return prediction

    def get_demand_per_community(
        self, time_of_day: int, day_of_week: int, month: int
    ) -> np.ndarray:
        """
        Get demand per community for the given time of day, day of week, and month.
        """
        zone_demand = self.get_demand_per_zone(time_of_day, day_of_week, month)

        community_demand = np.zeros(self.num_communities)
        for community in range(self.num_communities):
            zone_indices = np.where(self.zone_community_map[:, community] == 1)[0]
            community_demand[community] = np.sum(zone_demand[zone_indices])
        return community_demand

    def _maxminscaler_3d(self, tensor_3d: np.ndarray, data_range=(0, 1)):
        """
        Scales a 3D numpy array to a given range using global min/max. Like the implementation of the IrConv-LSTM model.
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

    def _get_target_index(self, time_of_day: int, day_of_week: int, month: int) -> int:
        # Filter index based on time components
        matching_timestamps = self.demand_data.index[
            (self.demand_data.index.hour == time_of_day)
            & (self.demand_data.index.dayofweek == day_of_week)
            & (self.demand_data.index.month == month)
        ]

        if matching_timestamps.empty:
            raise ValueError(
                f"No historical data found for time_of_day={time_of_day}, "
                f"day_of_week={day_of_week}, month={month} in the provided demand data."
            )

        current_dt = matching_timestamps[-1]  # Get the most recent matching timestamp

        # Get its integer position using get_indexer
        target_idx_array = self.demand_data.index.get_indexer([current_dt])

        return target_idx_array[0]

    def _get_historical_data_point(self, target_idx: int, offset: int) -> np.ndarray:
        """
        Retrieves a historical data point (num_zones, 1) from self.scaled_demand_data.
        Handles index out of bounds by returning zeros.
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
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # Convert numpy arrays to PyTorch tensors
        closeness_tensor = Variable(
            torch.FloatTensor(closeness_data_np).unsqueeze(0)
        ).to(device)
        period_tensor = Variable(torch.FloatTensor(period_data_np).unsqueeze(0)).to(
            device
        )
        trend_tensor = Variable(torch.FloatTensor(trend_data_np).unsqueeze(0)).to(
            device
        )

        with torch.no_grad():  # Disable gradient calculations for inference
            prediction_tensor = self.model(
                closeness_tensor, period_tensor, trend_tensor
            )

        # Squeeze to get (num_nodes,) for a single prediction item
        prediction_np = prediction_tensor.squeeze().cpu().numpy()

        return prediction_np

    def get_closeness_data(
        self, time_of_day: int, day_of_week: int, month: int
    ) -> np.ndarray:
        """
        Get closeness data for the given date.
        Output shape: (closeness_size, num_zones, 1)
        Order: T-closeness_size, ..., T-1
        """
        target_idx = self._get_target_index(time_of_day, day_of_week, month)

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

    def get_period_data(
        self, time_of_day: int, day_of_week: int, month: int
    ) -> np.ndarray:
        """
        Get period data for the given date.
        Output shape: (period_size, num_zones, 1)
        Order: T-(period_size*24h), ..., T-(1*24h)
        """
        target_idx = self._get_target_index(time_of_day, day_of_week, month)

        period_data_list = []
        for p_val in range(self.period_size, 0, -1):  # Iterates period_size, ..., 1
            offset = p_val * 24  # 24 hours in a day
            period_data_list.append(self._get_historical_data_point(target_idx, offset))

        if not period_data_list and self.period_size > 0:
            return np.zeros((self.period_size, self.num_zones, 1), dtype=np.float32)
        return np.array(period_data_list, dtype=np.float32)

    def get_trend_data(
        self, time_of_day: int, day_of_week: int, month: int
    ) -> np.ndarray:
        """
        Get trend data for the given date.
        Output shape: (trend_size, num_zones, 1)
        Order: T-(trend_size*168h), ..., T-(1*168h)
        """
        target_idx = self._get_target_index(time_of_day, day_of_week, month)

        trend_data_list = []
        for t_val in range(self.trend_size, 0, -1):  # Iterates trend_size, ..., 1
            offset = t_val * 168  # 168 hours in a week (7 days)
            trend_data_list.append(self._get_historical_data_point(target_idx, offset))

        if not trend_data_list and self.trend_size > 0:
            return np.zeros((self.trend_size, self.num_zones, 1), dtype=np.float32)
        return np.array(trend_data_list, dtype=np.float32)
