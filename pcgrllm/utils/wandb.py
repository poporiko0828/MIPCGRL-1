from copy import deepcopy

import wandb
from os import path
from conf.config import Config
from pcgrllm.utils.logger import get_wandb_name
# from pcgrllm.utils.path_utils import get_exp_name


def start_wandb(config: Config):
    config = deepcopy(config)

    if wandb.api.api_key:
        config.wandb_key = wandb.api.api_key

    if config.wandb_key and config.wandb_project:

        wandb.login(key=config.wandb_key)
        run = wandb.init(
            project=config.wandb_project,
            resume=config.wandb_resume,
            id=get_wandb_name(config),
            name=get_wandb_name(config),
            save_code=True)

        wandb.define_metric("Evaluation/llm_iteration")
        # define which metrics will be plotted against it
        wandb.define_metric("Evaluation/*", step_metric="Evaluation/llm_iteration")

        wandb.define_metric("train/step")
        wandb.define_metric("Iteration*", step_metric="train/step")

        wandb.config.update(dict(config), allow_val_change=True)




def finish_wandb():
    if wandb.run is not None:
        wandb.finish()