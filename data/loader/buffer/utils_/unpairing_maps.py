import numpy as np
from typing import Tuple


def unpairing_maps(pairs: np.ndarray, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    prev_env_map = pairs[:, 0:cols, :]

    curr_env_map = pairs[:, cols:, :]

    return prev_env_map, curr_env_map
