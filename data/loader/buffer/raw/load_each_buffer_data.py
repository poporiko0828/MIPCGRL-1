from typing import List
from pqdm.threads import pqdm

from flax.struct import dataclass
from numpy.random import choice
import numpy as np
from typing import Optional


from .utils_ import get_file_paths


@dataclass
class EachData:
    file_list: np.ndarray
    done_indices: List[np.ndarray]
    map_obs: List[np.ndarray]
    env_map: List[np.ndarray]


def load_each_buffer_data(
    buffer_dir: str,
    file_limit: Optional[int] = None,
    n_jobs: int = 4,
):
    file_list = get_file_paths(buffer_dir, file_limit)

    n_jobs = np.clip(n_jobs, 1, min(len(file_list), file_limit)).item()

    if file_limit >= 1:
        file_list = file_list[choice(len(file_list), file_limit, replace=False)]

    loaded_data: List[np.ndarray] = pqdm(
        file_list,
        lambda path: np.load(path, allow_pickle=True).get("buffer").item(),
        desc="Load the buffer data",
        n_jobs=n_jobs,
        exception_behaviour="immediate",
    )

    done_indices = [data.get("done") for data in loaded_data]
    map_obs = [data.get("obs").get("map_obs") for data in loaded_data]
    env_map = [data.get("env_map") for data in loaded_data]

    return EachData(
        file_list=file_list,
        done_indices=done_indices,
        env_map=env_map,
        map_obs=map_obs,
    )
