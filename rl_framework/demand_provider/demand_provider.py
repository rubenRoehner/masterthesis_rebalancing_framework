from abc import ABC, abstractmethod
import numpy as np


class DemandProvider(ABC):
    """
    Abstract base class for demand provider module.
    """

    def __init__(
        self, num_communities: int, num_zones: int, zone_community_map: np.ndarray
    ):
        self.num_communities = num_communities
        self.num_zones = num_zones

    @abstractmethod
    def get_demand_per_zone(self, time_of_day, day_of_week, month) -> np.ndarray:
        pass

    @abstractmethod
    def get_demand_per_community(self, time_of_day, day_of_week, month) -> np.ndarray:
        pass
