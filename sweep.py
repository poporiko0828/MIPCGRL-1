import hydra
import os
from os.path import basename
import logging

import jax
import wandb

from conf.config import SweepConfig

from train import main as train_main
from eval import main as evaluate_main

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

@hydra.main(version_base=None, config_path="./conf", config_name="sweep_pcgrl")
def main(config: SweepConfig):

    logger.info(f'Starting training with config: {config}...')
    train_main(config)
    logger.info('Training finished.')

    if config.wandb_key:
        wandb.finish()

    jax.clear_caches()

    config.n_envs = 100
    config.wandb_project = f'eval_{config.wandb_project}'
    logger.info(f'Starting evaluation with config: {config}...')
    evaluate_main(config)
    logger.info('Evaluation finished.')


if __name__ == "__main__":
    main()
