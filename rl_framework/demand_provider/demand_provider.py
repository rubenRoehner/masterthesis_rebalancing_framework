"""
demand_provider.py

Abstract base class for demand provider modules.
This module defines the interface for providing historical and real-time e-scooter demand data
across zones and communities for simulation and training purposes.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DemandProvider(ABC):
    """Abstract base class for demand provider modules.

    This class defines the interface for providing e-scooter demand data
    at different granularities (zone-level and community-level) for simulation
    and reinforcement learning training.
    """

    def __init__(
        self, num_communities: int, num_zones: int, zone_community_map: pd.DataFrame
    ) -> None:
        """Initialize the DemandProvider.

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
        self.demand_data: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    def get_random_start_time(
        self, max_steps: int, step_duration: timedelta
    ) -> datetime:
        """Get a random valid start time for simulation episodes.

        Args:
            max_steps: maximum number of steps in the episode
            step_duration: duration of each simulation step

        Returns:
            datetime: valid start time that allows for full episode duration

        Raises:
            NotImplementedError: if not implemented by subclass
        """
        pass

    @abstractmethod
    def get_demand_per_zone(self, time_of_day: int, day: int, month: int) -> np.ndarray:
        """Get demand values for all zones at specified time.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: demand values for each zone

        Raises:
            NotImplementedError: if not implemented by subclass
        """
        pass

    @abstractmethod
    def get_demand_per_community(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        """Get aggregated demand values for all communities at specified time.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year

        Returns:
            np.ndarray: aggregated demand values for each community

        Raises:
            NotImplementedError: if not implemented by subclass
        """
        pass

    @abstractmethod
    def get_demand_per_zone_community(
        self, time_of_day: int, day: int, month: int, community_id: str
    ) -> np.ndarray:
        """Get demand values for zones within a specific community.

        Args:
            time_of_day: hour of day (0-23)
            day: day of month
            month: month of year
            community_id: ID of the target community

        Returns:
            np.ndarray: demand values for zones in the specified community

        Raises:
            NotImplementedError: if not implemented by subclass
        """
        pass
