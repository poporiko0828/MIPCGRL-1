import numpy as np
from ..types_ import IndicedData

from .load_each_buffer_data import load_each_buffer_data
from ..utils_ import generate_pair_indices


def load_buffer_data(
    buffer_dir: str,
    file_limit: int = 0,
    n_jobs: int = 4,
):
    data = load_each_buffer_data(buffer_dir, file_limit, n_jobs)

    done_indices = np.concat(data.done_indices).ravel()
    map_obs = data.map_obs
    env_map = data.env_map

    prev_indices, curr_indices = generate_pair_indices(done_indices)

    done_indices = np.concat(done_indices, axis=None)
    map_obs = np.concat(map_obs, axis=0)
    env_map = np.concat(env_map, axis=0)

    map_obs = map_obs.reshape((-1, *map_obs.shape[-3:]))
    env_map = env_map.reshape((-1, *env_map.shape[-2:]))

    return IndicedData(
        map_obs=map_obs,
        env_map=env_map,
        prev_indices=prev_indices,
        curr_indices=curr_indices,
    )
