"""
demand_forecaster.py

Abstract base class for demand forecasting modules.
This module defines the interface for predicting e-scooter demand patterns
across zones and communities using various forecasting techniques.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class DemandForecaster(ABC):
    """Abstract base class for demand forecasting modules.

    This class defines the interface for predicting e-scooter demand
    at different granularities (zone-level and community-level) using
    various machine learning and statistical forecasting approaches.
    """

    def __init__(
        self, num_communities: int, num_zones: int, zone_community_map: pd.DataFrame
    ) -> None:
        """Initialize the DemandForecaster.

        Args:
            num_communities: total number of communities in the system
            num_zones: total number of zones across all communities
            zone_community_map: DataFrame mapping zones to their communities

        Returns:
            None

        Raises:
            None
        """
        self.num_communities = num_communities
        self.num_zones = num_zones
        self.zone_community_map = zone_community_map

    @abstractmethod
    def predict_demand_per_zone(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        """Predict demand values for all zones at specified time.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: predicted demand values for each zone

        Raises:
            NotImplementedError: if not implemented by subclass
        """
        pass

    @abstractmethod
    def predict_demand_per_community(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        """Predict aggregated demand values for all communities at specified time.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: predicted aggregated demand values for each community

        Raises:
            NotImplementedError: if not implemented by subclass
        """
        pass

    @abstractmethod
    def predict_demand_per_zone_community(
        self, time_of_day: int, day: int, month: int, community_id: str
    ) -> np.ndarray:
        """Predict demand values for zones within a specific community.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year
            community_id: ID of the target community

        Returns:
            np.ndarray: predicted demand values for zones in the specified community

        Raises:
            NotImplementedError: if not implemented by subclass
        """
        pass
