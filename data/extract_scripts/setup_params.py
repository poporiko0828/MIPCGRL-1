from flax.struct import dataclass

import math
import os
from os.path import join
from glob import glob

from pipe import Pipe
import numpy as np

from conf.config import BertTrainConfig

from .generate_filename import generate_filename


@dataclass
class Params:
    buffer_dir: str
    file_list: list[str]
    total_files: int

    buffer_filename: str
    target_filename: str

    buffer_filepath: str
    target_filepath: str


def setup_params(
    model_name: str,
    buffer_dir: str,
    config: BertTrainConfig,
    instruct: str,
):
    dataset_path = config.dataset_path
    buffer_ratio = config.buffer_ratio
    fine_tune = config.fine_tune
    use_prev = config.use_prev

    os.makedirs(dataset_path, exist_ok=True)
    print(f"Created dataset directory: {dataset_path}")

    target_filename = f"buffer-ratio-{buffer_ratio}.npz"
    print(f"Buffer filename: {target_filename}")

    buffer_dir = "/bi2907/pcgrl_buffer" if config.use_hpc else buffer_dir
    print(f"Using buffer directory: {buffer_dir}")

    file_list = (
        join(buffer_dir, "**", "*.npz")
        | Pipe(lambda pattern: glob(pattern, recursive=True))
        | Pipe(lambda paths: np.random.permutation(paths))
    )
    total_files = len(file_list)
    print(f"Found {total_files} total buffer files")

    n_buffer = math.floor(total_files * buffer_ratio)
    print(f"Will use {n_buffer} buffer files (ratio: {buffer_ratio})")

    if n_buffer >= 1:
        file_list = file_list[0:n_buffer]
        total_files = len(file_list)
        print(f"Selected {total_files} buffer files after applying ratio")
    else:
        print("Warning: No buffer files selected. Check buffer_ratio setting.")

    buffer_filename = (
        generate_filename(model_name, fine_tune, instruct, buffer_ratio, use_prev)
        | Pipe(lambda filename: f"{filename}.npz")
        | Pipe(lambda filename: join(dataset_path, filename))
    )
    print(f"Target filename: {buffer_filename}")

    buffer_filepath = join(dataset_path, buffer_filename)
    target_filepath = join(dataset_path, target_filename)

    return Params(
        buffer_dir=buffer_dir,
        file_list=file_list,
        total_files=total_files,
        buffer_filename=buffer_filename,
        target_filename=target_filename,
        buffer_filepath=buffer_filepath,
        target_filepath=target_filepath,
    )
