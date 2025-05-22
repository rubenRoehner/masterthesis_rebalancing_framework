from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DemandProvider(ABC):
    """
    Abstract base class for demand provider module.
    """

    def __init__(
        self, num_communities: int, num_zones: int, zone_community_map: pd.DataFrame
    ):
        self.num_communities = num_communities
        self.num_zones = num_zones
        self.zone_community_map = zone_community_map
        self.demand_data: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    def get_random_start_time(
        self, max_steps: int, step_duration: timedelta
    ) -> datetime:
        pass

    @abstractmethod
    def get_demand_per_zone(self, time_of_day: int, day: int, month: int) -> np.ndarray:
        pass

    @abstractmethod
    def get_demand_per_community(
        self, time_of_day: int, day: int, month: int
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_demand_per_zone_community(
        self, time_of_day: int, day: int, month: int, community_id: str
    ) -> np.ndarray:
        pass
