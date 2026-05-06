from glob import glob
import numpy as np
from chex import dataclass
from os.path import basename

from tqdm import tqdm


import logging
import os
from conf.config import RewardTrainConfig


log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


@dataclass
class Dataset:
    prev_map_obs: np.ndarray
    curr_map_obs: np.ndarray
    reward: np.ndarray

def create_dataset(config: RewardTrainConfig):

    file_list = glob(os.path.join(config.buffer_dir, '**', "*.npz"), recursive=True)
    # file_list += glob(os.path.join(config.buffer_dir, "**", "*.npy"), recursive=True)
    total_files = len(file_list)

    assert total_files > 0, f"No buffer files found in {config.buffer_dir}"

    n_buffer = config.n_buffer
    if n_buffer > 0:
        file_list = file_list[:n_buffer]
        logging.info(f"Using {n_buffer} of {total_files} buffer files")

    arr_prev_map_obs, arr_curr_map_obs, arr_prev_env_map, arr_curr_env_map = [], [], [], []
    rewards = []
    logging.info(f"Loading {len(file_list)} buffer files")

    for file in tqdm(file_list, desc="Loading buffer files"):
        data = np.load(file, allow_pickle=True).get('buffer').item()

        obs = data.get('obs')
        map_obs = np.array(obs.get('map_obs'))
        reward = data.get('reward')
        done = np.array(data.get('done'))
        env_map = np.array(data.get('env_map'))

        prev_map_obs = map_obs[:, 0:-1]
        curr_map_obs = map_obs[:, 1:]
        prev_env_map = env_map[:, 0:-1]
        curr_env_map = env_map[:, 1:]
        reward = reward[:, 1:]


        done = done[:, 1:]

        done_indices = np.where(done != True)


        prev_map_obs = prev_map_obs[done_indices[0], done_indices[1], ...]
        curr_map_obs = curr_map_obs[done_indices[0], done_indices[1], ...]
        prev_env_map = prev_env_map[done_indices[0], done_indices[1], ...]
        curr_env_map = curr_env_map[done_indices[0], done_indices[1], ...]
        reward = reward[done_indices[0], done_indices[1], ...]

        arr_curr_env_map.append(curr_env_map)
        arr_prev_env_map.append(prev_env_map)
        arr_curr_map_obs.append(curr_map_obs)
        arr_prev_map_obs.append(prev_map_obs)
        rewards.append(reward)

    # Concat
    curr_map_obs = np.concatenate(arr_curr_map_obs, axis=0)
    prev_map_obs = np.concatenate(arr_prev_map_obs, axis=0)
    curr_env_map = np.concatenate(arr_curr_env_map, axis=0)
    prev_env_map = np.concatenate(arr_prev_env_map, axis=0)
    reward = np.concatenate(rewards, axis=0)

    # reward = reward.clip(-1, 1)

    if config.zero_reward_ratio is not None:
        zero_indices = np.where((reward < 0.001) & (reward > -0.001))[0]
        zero_ratio = len(zero_indices) / len(reward) * 100

        n_keep = int(len(zero_indices) * config.zero_reward_ratio) # (111, )
        zero_indices_keep = np.random.choice(zero_indices, n_keep, replace=False)

        non_zero_indices = np.where(reward != 0)[0] # (111, )
        sample_indices = np.concatenate([non_zero_indices, zero_indices_keep], axis=0)

        non_zero_count = len(non_zero_indices)
        final_zero_count = len(zero_indices_keep)
        final_total_count = len(sample_indices)
        filtered_zero_ratio = final_zero_count / final_total_count * 100
        filtered_non_zero_ratio = 100 - filtered_zero_ratio

        logging.info(
            f"Initial dataset: {len(reward):,} samples. Zero reward samples: {len(zero_indices):,} ({zero_ratio:.2f}%).")
        logging.info(
            f"After filtering: Non-zero samples: {non_zero_count:,}, Kept zero samples: {final_zero_count:,}.")
        logging.info(
            f"Final dataset: {final_total_count:,} samples. Zero reward: {filtered_zero_ratio:.2f}%, Non-zero: {filtered_non_zero_ratio:.2f}%.")

        curr_map_obs = curr_map_obs[sample_indices]
        prev_map_obs = prev_map_obs[sample_indices]
        curr_env_map = curr_env_map[sample_indices]
        prev_env_map = prev_env_map[sample_indices]
        reward = reward[sample_indices]


    reward_raw = np.clip(reward, -1, 1)

    sample_size = curr_env_map.shape[0]

    logging.info(f"Loaded {sample_size:,} samples")

    batch_size = config.batch_size

    dataset = Dataset(
                      prev_map_obs=prev_map_obs,
                      curr_map_obs=curr_map_obs,
                      reward=reward)

    # shuffle dataset
    indices = np.arange(sample_size)
    np.random.shuffle(indices)

    return dataset


def split_dataset(database: Dataset, train_ratio: float = 0.8):
    """
    Splits the dataset into train and test sets.

    Args:
        database (Dataset): The full dataset containing observations, rewards, and done flags.
        train_ratio (float): Proportion of the data to use for training. Default is 0.8.

    Returns:
        Tuple[Dataset, Dataset]: Train and Test datasets.
    """
    total_size = database.curr_map_obs.shape[0]
    train_size = int(total_size * train_ratio)

    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Train Dataset
    train_dataset = Dataset(
        prev_map_obs=database.prev_map_obs[train_indices],
        curr_map_obs=database.curr_map_obs[train_indices],
        reward=database.reward[train_indices],
    )

    # Test Dataset
    test_dataset = Dataset(
        prev_map_obs=database.prev_map_obs[test_indices],
        curr_map_obs=database.curr_map_obs[test_indices],
        reward=database.reward[test_indices],
    )

    return train_dataset, test_dataset



def create_batches(dataset: Dataset, batch_size: int, augment: bool = False):

    num_samples = len(dataset.curr_map_obs)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        prev = dataset.prev_map_obs[batch_indices]
        curr = dataset.curr_map_obs[batch_indices]

        if augment:
            if np.random.rand() > 0.5:
                prev = np.flip(prev, axis=1)
                curr = np.flip(curr, axis=1)

            if np.random.rand() > 0.5:
                prev = np.flip(prev, axis=2)
                curr = np.flip(curr, axis=2)

        X = (prev, curr)
        y = dataset.reward[batch_indices]


        yield X, y

