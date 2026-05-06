from os.path import isfile
import numpy as np

from .load_indiced_data import load_indiced_data
from ..raw import load_buffer_data
from ..types_ import IndicedData


def load_memoized_buffer_data(
    buffer_dir: str,
    target_path: str,
    file_limit: int = 0,
    n_jobs: int = 4,
    ignore_file: bool = False,
) -> IndicedData:

    is_cached = isfile(target_path) and not ignore_file

    if not is_cached:
        data = load_buffer_data(
            buffer_dir,
            file_limit=file_limit,
            n_jobs=n_jobs,
        )

        dataset = {
            "map_obs": data.map_obs,
            "env_map": data.env_map,
            "prev_indices": data.prev_indices,
            "curr_indices": data.curr_indices,
        }

        np.savez(target_path, **dataset)
    else:
        data = load_indiced_data(target_path)

    return data
