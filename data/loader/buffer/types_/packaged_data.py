import numpy as np
from typing import Optional

from flax.struct import dataclass


@dataclass
class PackagedData:
    map_pair_indices: np.ndarray
    instruction_indices: np.ndarray
    map_obs: np.ndarray
    env_map: np.ndarray

    rewards: np.ndarray

    instructions: np.ndarray
    reward_enums: np.ndarray
    embeddings: np.ndarray
    conditions: np.ndarray

    input_ids: Optional[np.ndarray]
    attention_mask: Optional[np.ndarray]

    is_finetuned: bool
