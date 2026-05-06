from data.loader.buffer import load_memoized_buffer_data

from data.loader.buffer.utils_ import pairing_maps, extract_unique_pair_indices


def load_buffer(
    source_dir: str,
    cache_path: str,
    total_files: int,
    n_jobs: int = 4,
    ignore_file: bool = False,
):
    buffer_data = load_memoized_buffer_data(
        source_dir,
        cache_path,
        total_files,
        n_jobs,
        ignore_file,
    )

    environment_pairs = pairing_maps(
        buffer_data.env_map[buffer_data.prev_indices],
        buffer_data.env_map[buffer_data.curr_indices],
    )

    index_pairs = pairing_maps(
        buffer_data.prev_indices[..., None],
        buffer_data.curr_indices[..., None],
    )

    selected_pair_indices = extract_unique_pair_indices(environment_pairs)

    return buffer_data, index_pairs[selected_pair_indices]
