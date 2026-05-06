from flax.struct import dataclass
import numpy as np


@dataclass
class DenseData:
    prev_map_obs: np.ndarray
    curr_map_obs: np.ndarray
    prev_env_map: np.ndarray
    curr_env_map: np.ndarray
