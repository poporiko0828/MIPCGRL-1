import numpy as np
from flax.struct import dataclass


@dataclass
class IndicedData:
    map_obs: np.ndarray
    env_map: np.ndarray
    prev_indices: np.ndarray
    curr_indices: np.ndarray
