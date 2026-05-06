import numpy as np
from flax.struct import dataclass


@dataclass
class Duplicates:
    unique_pairs: np.ndarray
    unique_counts: np.ndarray
    duplicates_ratio: float
    num_unique_value: int
    num_total_value: int
    min_count: int
    max_count: int


def calculate_duplicates(pairs: np.ndarray) -> Duplicates:

    unique_pairs, unique_counts = np.unique(pairs, axis=0, return_counts=True)

    num_total_value = pairs.shape[0]
    num_unique_value = unique_pairs.shape[0]

    duplicates_ratio = (num_total_value - num_unique_value) / num_total_value

    min_count, max_count = np.min(unique_counts).item(), np.max(unique_counts).item()

    return Duplicates(
        unique_pairs=unique_pairs,
        unique_counts=unique_counts,
        duplicates_ratio=duplicates_ratio,
        num_unique_value=num_unique_value,
        num_total_value=num_total_value,
        min_count=min_count,
        max_count=max_count,
    )
