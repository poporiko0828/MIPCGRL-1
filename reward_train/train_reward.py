import datetime
import math
import os
from functools import partial
from os import PathLike
from os.path import basename
from typing import Any

import jax.debug
import pandas as pd
from pandas.core.window.doc import kwargs_scipy
from tqdm import tqdm
import wandb
import hydra
import logging
import shutil
import numpy as np

from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import jit
import jax.numpy as jnp
from jax.experimental.array_serialization.serialization import logger
import optax

from instruct_rl.reward_set import get_reward_batch
from pcgrllm.utils.logger import get_wandb_name
from reward_train.data_utils import create_dataset, split_dataset
from reward_train.model import apply_model
from reward_train.path_utils import (get_ckpt_dir, init_config)

from conf.config import BertTrainConfig, RewardTrainConfig
from reward_train.data_utils import create_batches
from reward_train.visualize import create_scatter_plot

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger('absl').setLevel(logging.ERROR)

@partial(jit, static_argnums=(3,))
def train_step(train_state: TrainState, X_batch, y_batch, is_train=True, rng=None):
    # Define compute_loss as a closure to include fine_tune
    def compute_loss(params, X_batch, y_class):
        X_prev_env_map, X_curr_env_map = X_batch  # Unpack input
        X_prev_env_map = jnp.expand_dims(X_prev_env_map, axis=0)
        X_curr_env_map = jnp.expand_dims(X_curr_env_map, axis=0)

        X_env_map = jnp.concatenate([X_prev_env_map, X_curr_env_map], axis=-1)

        predictions = train_state.apply_fn(params, X_env_map, train=is_train, rngs={'dropout': rng})

        logits = jax.nn.log_softmax(predictions)

        one_hot_labels = jax.nn.one_hot(y_class, num_classes=5)
        loss = -jnp.sum(one_hot_labels * logits, axis=-1)
        return jnp.mean(loss), logits.argmax(axis=-1)  # Return loss and predicted class

    # Vectorize over batch dimension
    compute_loss_vectorized = jax.vmap(
        compute_loss,
        in_axes=(None, 0, 0),  # Batch over X_batch and y_batch
        out_axes=(0, 0)  # Return loss and predictions per sample
    )

    # Compute gradients and loss
    def loss_fn(params):
        loss, predictions = compute_loss_vectorized(params, X_batch, y_batch)
        return jnp.mean(loss), predictions  # Return loss and predictions


    (mean_loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params
    )

    # Update weights if training
    train_state = jax.lax.cond(
        is_train,
        lambda _: train_state.apply_gradients(grads=grads),
        lambda _: train_state,
        operand=None,
    )

    return train_state, mean_loss, predictions  # Return state, loss, and predictions



def make_train(config: BertTrainConfig):
    def train(rng):
        database = create_dataset(config)

        reward_max, reward_min = np.max(database.reward), np.min(database.reward)

        train_set, test_set = split_dataset(database, train_ratio=config.train_ratio)
        train_state = get_train_state(config, rng)

        n_train = len(train_set.curr_map_obs)
        n_test = len(test_set.curr_map_obs)

        # Training loops
        n_train_batch = math.ceil(n_train / config.batch_size)
        n_test_batch = math.ceil(n_test / config.batch_size)

        all_df_train = pd.DataFrame()
        all_df_val = pd.DataFrame()

        for epoch in range(config.n_epochs):

            train_loss = 0
            i = 1
            val_loss = 0

            with tqdm(total=n_train_batch + n_test_batch, desc=f"Epoch {epoch + 1}") as pbar:
                # Training Loop

                train_y_gt, train_y_pd = list(), list()
                for X_batch, y_batch in create_batches(train_set, config.batch_size, augment=config.augment):
                    X_batch = jax.device_put(X_batch)
                    y_batch = jax.device_put(y_batch)

                    rng, _rng = jax.random.split(rng)

                    train_state, batch_loss, predictions = train_step(train_state, X_batch, y_batch, is_train=True, rng=_rng)
                    train_loss += batch_loss

                    train_y_gt.extend(y_batch), train_y_pd.extend(predictions)

                    pbar.update(1)
                    pbar.set_postfix({"Train Loss": train_loss / i, "Val Loss": val_loss})
                    i += 1

                train_loss /= (i - 1)

                # Validation Loop
                i = 1

                val_y_gt, val_y_pd = list(), list()
                for X_batch, y_batch in create_batches(test_set, config.batch_size):
                    X_batch = jax.device_put(X_batch)
                    y_batch = jax.device_put(y_batch)

                    _, batch_loss, predictions = train_step(train_state, X_batch, y_batch, is_train=False, rng=rng)
                    val_loss += batch_loss

                    val_y_gt.extend(y_batch), val_y_pd.extend(predictions)

                    pbar.update(1)
                    pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss / i})
                    i += 1

                val_loss /= (i - 1)

            rand_train_idx = np.random.choice(len(train_y_gt), 1000, replace=True)
            rand_val_idx = np.random.choice(len(val_y_gt), 1000, replace=True)

            df_train = pd.DataFrame({
                "epoch": epoch,
                "ground_truth": [float(train_y_gt[j]) for j in rand_train_idx],
                "prediction": [float(train_y_pd[j]) for j in rand_train_idx]
            })

            df_val = pd.DataFrame({
                "epoch": epoch,
                "ground_truth": [float(val_y_gt[j]) for j in rand_val_idx],
                "prediction": [float(val_y_pd[j]) for j in rand_val_idx]
            })

            # all_df_train = pd.concat([all_df_train, df_train])
            # all_df_val = pd.concat([all_df_val, df_val])

            settings = {'epoch': epoch, 'config': config, 'min_val': 0, 'max_val': config.n_epochs,
                        'xlim': (reward_min - 0.2, reward_max + 0.2), 'ylim': (reward_min - 0.2, reward_max + 0.2)
                        }
            train_fig_path = create_scatter_plot(df_train, postfix='_train', **settings)
            val_fig_path = create_scatter_plot(df_val, postfix='_val', **settings)

            train_table = wandb.Table(dataframe=df_train)
            val_table = wandb.Table(dataframe=df_val)

            if wandb.run is not None:
                wandb.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "epoch": epoch,
                    "train/prediction": train_table,
                    "val/prediction": val_table,
                    "train/result": wandb.Image(train_fig_path),
                    "val/result": wandb.Image(val_fig_path)
                })


            if wandb.run is not None:
                wandb.log({"train/loss": train_loss, "train/prediction": train_table,
                           "val/loss": val_loss, "val/prediction": val_table,
                           "epoch": epoch})


    return lambda rng: train(rng)



def get_train_state(config: BertTrainConfig, rng: jax.random.PRNGKey):
    def create_train_state(model, learning_rate, rng, num_samples, buffer=None, fine_tune=False, pretrained_model=None):
        params = model.init(rng, jnp.ones((num_samples, 31, 31, 14), dtype=jnp.float32), train=True)
        tx = optax.adamw(learning_rate, weight_decay=config.weight_decay)
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    model = apply_model(config=config)

    state = create_train_state(model,
                               learning_rate=config.lr,
                               rng=rng,
                               buffer=np.zeros((1, 31, 31, 7 * 2), dtype=np.float32),
                               num_samples=1)

    return state


def save_checkpoint(config, state, step):
    ckpt_dir = get_ckpt_dir(config)
    ckpt_dir = os.path.abspath(ckpt_dir)
    checkpoints.save_checkpoint(ckpt_dir, target=state, prefix="", step=step, overwrite=True, keep=3)
    print(f"Checkpoint saved at step {step}")


@hydra.main(version_base=None, config_path='../conf', config_name='train_reward')
def main(config: RewardTrainConfig):

    config = init_config(config)

    rng = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)

    if config.wandb_key:
        dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # wandb_id = f'{get_wandb_name(config)}-{dt}'
        wandb.login(key=config.wandb_key)
        wandb.init(
            name=config.exp_name,
            project=config.wandb_project,
            group=config.instruct,
            entity=config.wandb_entity,
            # name=get_wandb_name(config),
            # id=wandb_id,
            save_code=True)
        wandb.config.update(dict(config), allow_val_change=True)

    exp_dir = config.exp_dir
    logger.info(f'jax devices: {jax.devices()}')
    logger.info(f'running experiment at {exp_dir}')

    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)


    os.makedirs(exp_dir, exist_ok=True)

    make_train(config)(rng)

    wandb.finish()


if __name__ == '__main__':
    main()
