import numpy as np


def get_unique_pair_indices(pairs: np.ndarray) -> np.ndarray:

    _, unique_indices = np.unique(pairs, axis=0, return_index=True)

    return unique_indices
