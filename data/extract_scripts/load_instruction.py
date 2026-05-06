import numpy as np

from data.loader.instruction.load_instruction import Instruction
from data.loader.instruction import (
    load_instruction as load_instruction_,
)


def load_instruction(
    csv_paths: list[str], n_job: int
) -> tuple[Instruction, np.ndarray]:
    data = load_instruction_(
        csv_paths,
        n_job,
    )

    indices = np.arange(data.instructions.shape[0])

    return data, indices
