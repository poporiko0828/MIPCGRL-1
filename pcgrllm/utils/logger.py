import logging
from os import path
import numpy as np
import wandb
from glob import glob

from conf.config import Config
from pcgrllm.evaluation.base import EvaluationResult


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def print_log(logger: logging.Logger, message: str, level: int = logging.DEBUG):
    # divide to multiple lines
    for line in message.split('\n'):
        logger.log(level, line)


def get_group_name(config):
    group_name = f"rep-{config.representation}_model-{config.model}"
    if config.embed_type != "test":
        group_name = group_name + f"_embed-{config.embed_type}"
    if config.encoder.model is not None:
        group_name = group_name + f"_enc-{config.encoder.model}_enctr-{str(config.encoder.trainable).lower()}"
    if config.instruct is not None:
        group_name = group_name + f"_inst-{config.instruct}"

    # RQ4 parameters
    if config.encoder.buffer_ratio != 1.0:
        group_name = group_name + f"_br-{config.encoder.buffer_ratio}"
    if config.encoder.output_dim != 64:
        group_name = group_name + f"_es-{config.encoder.output_dim}"

    return group_name


def get_wandb_name(config: Config):
    exp_dir_path = config.exp_dir
    # split by directory
    exp_dirs = exp_dir_path.split('/')

    return exp_dirs[-1]


def get_wandb_name_eval(config: Config):
    wandb_name = get_wandb_name(config)
    eval_dir = config.eval_dir.split('/')[-1]
    eval_dir = eval_dir.replace('eval_', '')
    wandb_eval_name = f"{wandb_name}--{eval_dir}"
    return wandb_eval_name


def text_to_html(text):
    html_text = text.replace('\n', '<br>')

    html_text = html_text.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')

    return html_text


def log_reward_generation_data(logger, target_path: str, iteration: int, name: str = "reward_generation"):
    if wandb.run is None: return None

    # get the image files
    json_files = glob(path.join(target_path, '*.json'))
    python_files = glob(path.join(target_path, '*.py'))

    for idx, items in enumerate(zip(json_files, python_files)):
        # log the json file
        wandb.log({f'Iteration_{iteration}/{name}/json': wandb.Html(open(items[0], 'r').read())})
        wandb.log({f'Iteration_{iteration}/{name}/code': wandb.Html(text_to_html(open(items[1], 'r').read()))})

    # Log the count of json files using logger


def log_rollout_data(logger, target_path: str, iteration: int):
    if wandb.run is None: return None

    # get the images and numpy dir
    image_dir = path.join(target_path, 'images')
    numpy_dir = path.join(target_path, 'numpy')

    # get the image files
    image_files = glob(path.join(image_dir, '*.png'))
    numpy_files = glob(path.join(numpy_dir, '*.npy'))

    # log the images
    for idx, image_file in enumerate(image_files):
        # turn off wandb error

        wandb.log({f'Iteration_{iteration}/rollout/images': wandb.Image(image_file)})

    # log the numpy files, load as uint16 and log as string
    for idx, numpy_file in enumerate(numpy_files):
        numpy_data = np.load(numpy_file).astype(np.uint16)

        # make numpy string
        numpy_data = np.array2string(numpy_data, separator=',', max_line_width=10000)
        wandb.log({f'Iteration_{iteration}/rollout/numpy': wandb.Html(numpy_data)})

    # Log the count of images and numpy files using logger
    logger.info(f"Logged {len(image_files)} image files and {len(numpy_files)} numpy files to wandb for rollout.")


def log_feedback_data(logger, target_path: str, iteration: int):
    if wandb.run is None: return None

    # read json file
    json_files = glob(path.join(target_path, '*.json'))

    if len(json_files) > 0:
        json_file = json_files[0]
        if path.basename(json_file).startswith('feedback_log'):
            wandb.log({f'Iteration_{iteration}/feedback/context': wandb.Html(open(json_file, 'r').read())})

    text_files = glob(path.join(target_path, '*.txt'))

    if len(text_files) > 0:
        text_file = text_files[0]
        if path.basename(text_file) == 'feedback.txt':
            wandb.log({f'Iteration_{iteration}/feedback/response': wandb.Html(open(text_file, 'r').read())})

    # Log the count of json and text files using logger
    logger.info(f"Logged {len(json_files)} json files and {len(text_files)} text files to wandb for feedback.")


def log_evaluation_result(logger, result: EvaluationResult, iteration: int, evaluator_type: str):
    if wandb.run is None: return None

    result_dict = result.to_dict()

    for key, value in result_dict.items():
        if evaluator_type:
            log_key = f'Evaluation/{evaluator_type}/{key}'
        else:
            log_key = f'Evaluation/{key}'

        wandb.log({
            f'{log_key}': value,
            f'Evaluation/llm_iteration': iteration
        })

    # Log the evaluation result using logger
    logger.info(f"Logged evaluation result to wandb for iteration {iteration}.")
