import numpy as np


def pairing_maps(prev_env_map: np.ndarray, curr_env_map: np.ndarray):
    return np.concatenate([prev_env_map, curr_env_map], axis=1)
