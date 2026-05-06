import itertools
import logging
import os
import shutil
from os.path import dirname, isfile, join, abspath
from typing import List, Optional, Tuple, Union
from pipe import Pipe, batched, tee
import psutil
import threading
import time
from collections import defaultdict
import traceback

import flax.linen as nn
import hydra
import jax
import numpy as np
import scipy.stats as stats
from pqdm.threads import pqdm

from conf.config import BertTrainConfig
from data.loader.buffer.abstracted import load_packaged_data
from instruct_rl.reward_set import get_reward_batch
from LLM.LLM_utils import *
from LLM.path_utils import init_config

from .setup_params import setup_params
from .logger import logger

from .load_buffer import load_buffer
from .load_instruction import load_instruction

n_job_for_reading = 16
n_job_for_processing = 16
n_job_for_massive_processing = 16

prev_map_obs: Optional[np.ndarray] = None
curr_map_obs: Optional[np.ndarray] = None
prev_env_map: Optional[np.ndarray] = None
curr_env_map: Optional[np.ndarray] = None


def create_normalize_dataset(
    model_name: str,
    buffer_dir: str,
    config: BertTrainConfig,
    tokenizer: nn.Module,
    instruct: str,
    csv_paths: List[str],
    n_batch: int = 1_024,
) -> dict:
    params_ = setup_params(
        model_name,
        buffer_dir,
        config,
        instruct,
    )

    buffer_dir = params_.buffer_dir
    total_files = params_.total_files

    buffer_filepath = params_.buffer_filepath
    target_filepath = params_.target_filepath

    if not isfile(target_filepath):
        # logger.info(f"Loading {file_list} ")
        logger.info(f"target_filepath {target_filepath}")

        buffer, buffer_pair_indices = load_buffer(
            buffer_dir,
            buffer_filepath,
            total_files,
            n_jobs=n_job_for_reading,
            ignore_file=True,
        )

        print(f"buffer_pair_indices: {buffer_pair_indices}")

        instruction, instruction_indices = load_instruction(
            csv_paths, n_job_for_reading
        )
        conditions = instruction.conditions
        embeddings = instruction.embeddings
        reward_enums = instruction.reward_enums
        instructions = instruction.instructions

        unique_buffer_instruction_pair_indices = (
            itertools.product(buffer_pair_indices, instruction_indices)
            | Pipe(list)
            | Pipe(np.array)
            | Pipe(
                lambda pair_indices: pair_indices[
                    np.random.permutation(len(pair_indices))
                ]
            )
        )

        pair_indices = unique_buffer_instruction_pair_indices[:, 0]
        instruction_indices = unique_buffer_instruction_pair_indices[:, 1]

        if config.fine_tune:
            instructions = instruction.instructions

            inst_token = tokenizer(
                instructions.squeeze().tolist(),
                return_tensors="jax",
                padding="max_length",
                max_length=config.max_length,
                truncation=True,
            )

            input_ids, attention_mask = (
                np.array(inst_token["input_ids"]),
                np.array(inst_token["attention_mask"]),
            )
        else:
            input_ids, attention_mask = None, None

        print(f"instruction_indices.shape: {instruction_indices.shape}")
        print(f"pair_indices.shape: {pair_indices.shape}")

        exit(-1)

        rewards = get_reward_batch(
            reward_enums[instruction_indices],
            conditions[instruction_indices],
            buffer.env_map[pair_indices],
            buffer.env_map[pair_indices],
        )

        print(rewards.shape)

        dataset_dict = {
            "map_pair_indices": pair_indices,
            "instruction_indices": instruction_indices,
            "env_map": buffer.env_map,
            "map_obs": buffer.map_obs,
            "instructions": instructions,
            "reward_enums": reward_enums,
            "embeddings": embeddings,
            "conditions": conditions,
            "rewards": rewards,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "is_finetuned": config.fine_tune,
        }

        np.savez(target_filepath, **dataset_dict)
    else:
        logger.info(f"Loading {target_filepath}")

        dataset_dict = load_packaged_data(target_filepath)

    return dataset_dict


def check_reward_distribution(rewards: np.ndarray) -> np.ndarray:
    unique_reward = np.sort(np.unique(rewards))
    dist = np.zeros((unique_reward.shape[0], 2))

    for i, reward_value in enumerate(unique_reward):
        indices = np.where(rewards == reward_value)[0]
        num_reward = indices.shape[0]
        dist[i][0], dist[i][1] = reward_value, num_reward

    return dist


def norm_dist_indices(
    rewards: np.ndarray,
    pair_size: int,
    fine_tune=False,
) -> np.ndarray:
    reward_array = np.asarray(rewards)

    filtered_reward = reward_array[reward_array != 0]

    mean_value, std_value = np.mean(filtered_reward), np.std(filtered_reward)
    if std_value == 0:
        std_value = np.array(1.0)

    unique_reward = np.sort(np.unique(np.append(filtered_reward, [0])))

    pdf_values = stats.norm.pdf(unique_reward, mean_value, std_value)

    scaled_p_values = np.round(pdf_values * len(reward_array)).astype(int)

    def normalize(params: Tuple[np.ndarray, np.ndarray]):
        reward_value, p_scaled = params

        indices = np.where(reward_array == reward_value)[0]

        if len(indices) > p_scaled:
            sampled_indices = np.random.choice(indices, p_scaled, replace=False)
        else:
            repeat_factor = p_scaled // len(indices)
            remainder = p_scaled % len(indices)

            sampled_indices = np.concatenate(
                [
                    np.take_along_axis(
                        indices,
                        np.arange(indices.shape[0] * repeat_factor) % indices.shape[0],
                        0,
                    ),
                    np.random.choice(indices, remainder, replace=False),
                ]
            )

        if sampled_indices.shape[0] <= 0:
            return None

        return sampled_indices

    new_indices: List[Union[np.ndarray, None]] = pqdm(
        list(zip(unique_reward, scaled_p_values)),
        normalize,
        n_jobs=n_job_for_massive_processing,
        desc="calculating normalized distribution",
    )
    cleaned_new_indices: List[np.ndarray] = [
        indices for indices in new_indices if isinstance(indices, np.ndarray)
    ]
    norm_indices = np.concatenate(cleaned_new_indices)
    norm_indices = norm_indices[
        np.random.permutation(norm_indices.shape[0])[:pair_size]
    ]

    return norm_indices


def get_tokenizer(config: BertTrainConfig) -> Optional[nn.Module]:
    if config.fine_tune:
        pretrained_model, tokenizer = apply_pretrained_model(config=config)
    else:
        pretrained_model, tokenizer = None, None

    return tokenizer


@hydra.main(
    version_base=None,
    config_path=abspath(join(dirname(__file__), "..", "..", "conf")),
    config_name="train_bert",
)
def main(config: BertTrainConfig):
    config = init_config(config)

    np.random.seed(config.seed)

    exp_dir = config.exp_dir
    logger.info(f"jax devices: {jax.devices()}")
    logger.info(f"running experiment at {exp_dir}")

    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    tokenizer = get_tokenizer(config)

    create_normalize_dataset(
        config.pretrained_model,
        config.buffer_path,
        config,
        tokenizer,
        config.instruct,
        [
            join(
                dirname(__file__),
                "instruct",
                f"test/{config.pretrained_model}",
                f"{config.instruct}.csv",
            )
        ],
    )


stop_monitoring = False


if __name__ == "__main__":

    stats = defaultdict(list)
    stop_monitoring = False

    def monitor_resources():
        global stop_monitoring

        process = psutil.Process()
        while not stop_monitoring:
            stats["memory_mb"].append(
                process.memory_info().rss / (1024 * 1024)
            )
            stats["cpu_percent"].append(process.cpu_percent(interval=0.1))
            time.sleep(0.1)

    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.start()

    try:
        result = main()
    except Exception as _:
        print(traceback.format_exc())

        stop_monitoring = True
        monitor_thread.join()

    stop_monitoring = True
    monitor_thread.join()

