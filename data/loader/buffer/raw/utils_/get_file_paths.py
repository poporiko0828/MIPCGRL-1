from glob import glob
from os.path import join
import numpy as np
from pipe import Pipe
from typing import Optional
import logging


def get_file_paths(buffer_dir: str, limited_to: Optional[int] = None):
    file_list = glob(join(buffer_dir, "**", "*.npz"), recursive=True) | Pipe(np.array)

    n_sample = limited_to if isinstance(limited_to, int) else len(file_list)

    np.random.shuffle(file_list)

    return file_list[:n_sample]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    TEST_DIR = "./pcgrl_buffer"

    N_RANDOMIZE_TEST = 3
    N_FILE_LIMIT = 10

    logging.info("Start randomize test")

    path_for_all_samples = [get_file_paths(TEST_DIR) for _ in range(N_RANDOMIZE_TEST)]

    path_for_limited_samples = [
        get_file_paths(TEST_DIR, limited_to=N_FILE_LIMIT)
        for _ in range(N_RANDOMIZE_TEST)
    ]

    all_samples_identical = []
    for i in range(N_RANDOMIZE_TEST):
        for j in range(i + 1, N_RANDOMIZE_TEST):
            identical = np.array_equal(path_for_all_samples[i], path_for_all_samples[j])
            all_samples_identical.append(identical)

    limited_samples_identical = []
    for i in range(N_RANDOMIZE_TEST):
        for j in range(i + 1, N_RANDOMIZE_TEST):
            identical = np.array_equal(
                path_for_limited_samples[i], path_for_limited_samples[j]
            )
            limited_samples_identical.append(identical)

    all_test_passed = not any(all_samples_identical)
    limited_test_passed = not any(limited_samples_identical)

    if not all_test_passed:
        identical_count = sum(all_samples_identical)
        total_comparisons = len(all_samples_identical)

    if not limited_test_passed:
        identical_count = sum(limited_samples_identical)
        total_comparisons = len(limited_samples_identical)

    for test_name, samples in [
        ("whole file", path_for_all_samples),
        ("limited file", path_for_limited_samples),
    ]:
        if len(samples) >= 2:
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    set_i = set(samples[i])
                    set_j = set(samples[j])
                    common_elements = set_i & set_j
                    if test_name == "whole file":
                        logging.info(
                            f"{test_name} test: Implement {i + 1} and {j + 1} are {'difference' if not np.array_equal(samples[i], samples[j]) else 'equal'}"
                        )
                    else:
                        overlap_percentage = (
                            len(common_elements) / len(set_i) * 100
                            if len(set_i) > 0
                            else 0
                        )
                        logging.info(
                            f"{test_name} test: Implement {i + 1} and {j + 1} duplicate rate {overlap_percentage:.1f}%"
                        )
