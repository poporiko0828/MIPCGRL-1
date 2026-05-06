from pqdm.threads import pqdm
from tqdm import tqdm

from flax.struct import dataclass

from typing import Iterable, List
import polars
import numpy as np


@dataclass
class Instruction:

    instructions: np.ndarray
    reward_enums: np.ndarray
    embeddings: np.ndarray
    conditions: np.ndarray


def load_instruction(
    instruction_paths: Iterable[str],
    n_jobs: int = 4,
) -> Instruction:
    n_jobs = np.clip(n_jobs, 1, len(instruction_paths)).item()

    dfs: List[polars.DataFrame] = pqdm(
        instruction_paths,
        lambda path: polars.read_csv(
            path
        ),
        n_jobs=n_jobs,
    )

    instructions: np.ndarray = np.concatenate(
        [df[["instruction"]].to_numpy() for df in dfs], axis=0
    )

    reward_enums: np.ndarray = np.concatenate(
        [df[["reward_enum"]].to_numpy() for df in dfs], axis=0
    )

    embeddings: np.ndarray = np.concatenate(
        [
            df.select(
                polars.col(r"^embed_.*$")
            ).to_numpy()
            for df in tqdm(
                dfs, desc="Extracting embeddings"
            )
        ],
        axis=0,
    )

    conditions: np.ndarray = np.concatenate(
        [
            df.select(
                polars.col(r"^condition_.*$")
            ).to_numpy()
            for df in tqdm(
                dfs, desc="Extracting conditions"
            )
        ],
        axis=0,
    )

    del dfs

    return Instruction(
        instructions=instructions,
        reward_enums=reward_enums,
        embeddings=embeddings,
        conditions=conditions,
    )
