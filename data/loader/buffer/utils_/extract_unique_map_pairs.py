import numpy as np
from .extract_unique_pair_indices import extract_unique_pair_indices
from .pairing_maps import pairing_maps
from .unpairing_maps import unpairing_maps
from ..types_ import IndicedData


def extract_unique_map_pairs(
    data: IndicedData,
    cols: int = 16,
) -> np.ndarray:

    map_obs = data.map_obs
    env_map = data.env_map
    prev_indices = data.prev_indices
    curr_indices = data.curr_indices

    map_pairs = pairing_maps(
        env_map[prev_indices],
        env_map[curr_indices],
    )
    map_pairs_indices = pairing_maps(
        prev_indices[..., None],
        curr_indices[..., None],
    )

    unique_map_pairs, unique_map_pair_indices = np.unique(
        map_pairs, axis=0, return_index=True
    )

    del map_pairs