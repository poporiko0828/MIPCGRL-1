import numpy as np


def validate_pairing(
    prev_env_map: np.ndarray,
    curr_env_map: np.ndarray,
    pairs: np.ndarray,
):
    return np.all(
        np.array(
            [
                np.all(np.concatenate([prev_map, curr_map]) == pairs[i])
                for i, (prev_map, curr_map) in enumerate(
                    zip(prev_env_map, curr_env_map)
                )
            ]
        )
    )
