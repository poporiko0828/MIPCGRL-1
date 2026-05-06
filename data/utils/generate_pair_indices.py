import numpy as np


def generate_pair_indices(done_indices: np.ndarray):


    if done_indices.ndim != 1:
        raise TypeError(
            f"done_indices must be 1-dimension array. Current dimension: {done_indices.ndim}"
        )

    undone_indices = np.argwhere(done_indices != True).ravel()
    done_indices = np.argwhere(done_indices == True).ravel()

    prev_indices = np.argwhere(undone_indices - 1 >= 0).squeeze()

    curr_indices = (prev_indices + 1).squeeze()

    return prev_indices, curr_indices


if __name__ == "__main__":
    sample_indices = np.array([False, False, False, False, True, False, False, False])
    print(generate_pair_indices(sample_indices))
