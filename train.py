import argparse
from datetime import datetime
import logging
import os
from glob import glob

import wandb
import shutil
from functools import partial
from os.path import abspath, basename, dirname, join
from timeit import default_timer as timer
from typing import Any, NamedTuple, Tuple
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pandas as pd
from flax.training.train_state import TrainState
from jax.experimental.array_serialization.serialization import logger
from tensorboardX import SummaryWriter

from conf.config import Config, TrainConfig
from envs.pcgrl_env import (
    OldQueuedState,
    gen_dummy_queued_state,
    gen_dummy_queued_state_old,
)
from instruct_rl.dataclass import Instruct
from instruct_rl.evaluate import get_loss_batch
from instruct_rl.reward_set import get_reward_batch
from instruct_rl.reward_set_llm import get_reward_batch_llm
from pcgrllm.utils.log_handler import (
    CSVLoggingHandler,
    MultipleLoggingHandler,
    TensorBoardLoggingHandler,
    WandbLoggingHandler,
)
from pcgrllm.utils.logger import get_wandb_name, get_group_name
from pcgrllm.utils.path_utils import (
    get_ckpt_dir,
    gymnax_pcgrl_make,
    init_config,
    init_network,
)
from purejaxrl.experimental.s5.wrappers import LLMRewardWrapper, LogWrapper
from purejaxrl.structures import RunnerState, Transition

log_level = os.getenv(
    "LOG_LEVEL", "INFO"
).upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
from utils import make_sim_render_episode_single, render_callback

logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


def log_callback(metric, steps_prev_complete, config, writer, train_start_time):
    timesteps = metric["timestep"][metric["returned_episode"]] * config.n_envs
    return_values = metric["returned_episode_returns"][metric["returned_episode"]]

    if len(timesteps) > 0:
        t = timesteps[-1].item()
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()

        ep_length = metric["returned_episode_lengths"][
            metric["returned_episode"]
        ].mean()
        fps = (t - steps_prev_complete) / (timer() - train_start_time)

        prefix = (
            f"Iteration_{config.current_iteration}/train/"
            if config.current_iteration > 0
            else ""
        )

        metric = {
            f"Train/{prefix}ep_return": ep_return_mean,
            f"Train/{prefix}ep_return_max": ep_return_max,
            f"Train/{prefix}ep_return_min": ep_return_min,
            f"Train/{prefix}ep_length": ep_length,
            f"Train/{prefix}fps": fps,
            f"Train/Step": t,
        }

        # log metrics
        writer.log(metric, t)

        print(
            f"[train] global step={t}; episodic return mean: {ep_return_mean} "
            + f"max: {ep_return_max}, min: {ep_return_min}, fps: {fps}"
        )


def eval_callback(
    eval_metric,
    train_metric,
    states,
    frames,
    steps_prev_complete,
    config,
    writer,
    train_start_time,
):
    timesteps = (
        train_metric["timestep"][train_metric["returned_episode"]] * config.n_envs
    )
    return_values = eval_metric["returned_episode_returns"][
        eval_metric["returned_episode"]
    ]

    # jax.debug.print("{}", return_values)

    if len(timesteps) > 0:
        t = timesteps[-1].item()
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()

        ep_length = eval_metric["returned_episode_lengths"][
            eval_metric["returned_episode"]
        ].mean()
        prefix = (
            f"Iteration_{config.current_iteration}/train/"
            if config.current_iteration > 0
            else ""
        )

        metric = {
            f"Eval/{prefix}ep_return": ep_return_mean,
            f"Eval/{prefix}ep_return_max": ep_return_max,
            f"Eval/{prefix}ep_return_min": ep_return_min,
            f"Eval/{prefix}ep_length": ep_length,
            "Train/Step": t,
        }

        # log metrics
        writer.log(metric, t)
        render_callback(
            frames=frames,
            states=states,
            video_dir=config._vid_dir,
            image_dir=config._img_dir,
            numpy_dir=config._numpy_dir,
            logger=logger,
            config=config,
            t=t,
        )

        print(
            f"[eval] global step={t}; episodic return mean: {ep_return_mean} "
            + f"max: {ep_return_max}, min: {ep_return_min}"
        )


def loss_callback(metric, loss, config, writer):
    timesteps = metric["timestep"][metric["returned_episode"]] * config.n_envs

    if len(timesteps) > 0:
        t = timesteps[-1].item()

        result_df = pd.DataFrame({"reward_enum": loss.reward_enum, "loss": loss.loss})

        mean_loss = result_df.groupby("reward_enum").agg({"loss": ["mean"]})
        mean_loss.columns = mean_loss.columns.droplevel(0)
        mean_loss = mean_loss.reset_index()

        dict_loss = dict()
        for _, row in mean_loss.iterrows():
            reward_enum, mean = row
            dict_loss[f"Loss/{str(int(reward_enum))}"] = mean

        writer.log(dict_loss, t)
        dict_str = ", ".join([f"{k}: {v}" for k, v in dict_loss.items()])
        print(f"[eval] global step={t}; loss: {dict_str}")


def make_train(config, restored_ckpt, checkpoint_manager, encoder_params):
    config.NUM_UPDATES = config.total_timesteps // config.num_steps // config.n_envs
    config.MINIBATCH_SIZE = config.n_envs * config.num_steps // config.NUM_MINIBATCHES

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)

    latest_update_step = checkpoint_manager.latest_step()
    if latest_update_step is None:
        latest_update_step = 0

    env = LogWrapper(env)
    env.init_graphics()

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.NUM_MINIBATCHES * config.update_epochs))
            / config.NUM_UPDATES
        )
        return config["LR"] * frac

    def train(rng, runner_state):
        train_start_time = timer()

        # Create a tensorboard writer
        writer = SummaryWriter(config.exp_dir)

        # INIT NETWORK
        network = init_network(env, env_params, config)

        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)

        network_params = network.init(_rng, init_x)

        if config.instruct_csv:
            csv_path = abspath(
                join(dirname(__file__), "instruct", f"{config.instruct_csv}.csv")
            )

            instruct_df = pd.read_csv(csv_path)

            def get_train_test(df, is_train=True):
                df = df[df["train"] == is_train]

                embedding_df = df.filter(regex="embed_*")
                embedding_df = embedding_df.reindex(
                    sorted(embedding_df.columns, key=lambda x: int(x.split("_")[-1])),
                    axis=1,
                )
                embedding = jnp.array(embedding_df.to_numpy())

                if config.nlp_input_dim > embedding.shape[1]:
                    embedding = jnp.pad(
                        embedding,
                        ((0, 0), (0, config.nlp_input_dim - embedding.shape[1])),
                        mode="constant",
                    )

                condition_df = df.filter(regex="condition_*")
                condition_df = condition_df.reindex(
                    sorted(condition_df.columns, key=lambda x: int(x.split("_")[-1])),
                    axis=1,
                )
                condition = jnp.array(condition_df.to_numpy())

                reward_enum_list = [[int(digit) for digit in str(num)] for num in instruct_df["reward_enum"].to_list()]

                max_len = max(len(x) for x in reward_enum_list)

                reward_enum = jnp.array([
                    x + [0] * (max_len - len(x)) for x in reward_enum_list
                ])

                return Instruct(
                    reward_i=reward_enum,
                    condition=condition,
                    embedding=embedding,
                )

            train_inst = get_train_test(instruct_df, is_train=True)
            test_inst = get_train_test(instruct_df, is_train=False)

        else:
            train_inst, test_inst = None, None

        if config.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV FOR TRAIN
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.n_envs)

        dummy_queued_state = gen_dummy_queued_state(env)

        # Apply pmap
        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        obsv, env_state = vmap_reset_fn(reset_rng, env_params, dummy_queued_state)

        rng, _rng = jax.random.split(rng)

        steps_prev_complete = 0
        runner_state = RunnerState(train_state, env_state, obsv, rng, update_i=0)

        if restored_ckpt is not None:
            steps_prev_complete = restored_ckpt["steps_prev_complete"]
            runner_state = restored_ckpt["runner_state"]
            steps_remaining = config.total_timesteps - steps_prev_complete
            config.NUM_UPDATES = int(
                steps_remaining // config.num_steps // config.n_envs
            )

        if encoder_params is not None:
            logger.info(
                f"Parameters loaded from encoder checkpoint ({config.encoder.ckpt_path})"
            )
            runner_state.train_state.params["params"]["subnet"]["encoder"] = (
                encoder_params
            )

        handler_classes = [
            TensorBoardLoggingHandler,
            WandbLoggingHandler,
            CSVLoggingHandler,
        ]
        multiple_handler = MultipleLoggingHandler(
            config=config, handler_classes=handler_classes, logger=logger
        )

        # Set the start time and previous steps
        multiple_handler.set_start_time(train_start_time)
        multiple_handler.set_steps_prev_complete(steps_prev_complete)

        multiple_handler.add_text("Train/Config", f"```{str(config)}```")
        # if reward_function is in this scope

        _log_callback = partial(
            log_callback,
            config=config,
            writer=multiple_handler,
            train_start_time=train_start_time,
            steps_prev_complete=steps_prev_complete,
        )

        # FIXME: Temporary hack for reloading binary after change to
        #   agent_coords generation.
        if config.representation == "narrow":
            runner_state = runner_state.replace(
                env_state=runner_state.env_state.replace(
                    env_state=runner_state.env_state.env_state.replace(
                        rep_state=runner_state.env_state.env_state.rep_state.replace(
                            agent_coords=runner_state.env_state.env_state.rep_state.agent_coords[
                                :, : config.map_width**2
                            ]
                        )
                    )
                )
            )

        def init_checkpoint(runner_state):
            ckpt = {"runner_state": runner_state, "step_i": 0}
            checkpoint_manager.save(0, args=ocp.args.StandardSave(ckpt))

        def save_checkpoint(runner_state, info, steps_prev_complete):
            # Get the global env timestep numbers corresponding to the points at which different episodes were finished
            timesteps = info["timestep"][info["returned_episode"]] * config.n_envs

            if len(timesteps) > 0:
                # Get the latest global timestep at which some episode was finished
                t = timesteps[-1].item()
                latest_ckpt_step = checkpoint_manager.latest_step()
                if latest_ckpt_step is None or t - latest_ckpt_step >= config.ckpt_freq:
                    print(f"Saving checkpoint at step {t}")
                    ckpt = {"runner_state": runner_state, "step_i": t}

                    checkpoint_manager.save(t, args=ocp.args.StandardSave(ckpt))

        # TRAIN LOOP
        def _update_step_with_render(update_runner_state, _):
            # COLLECT TRAJECTORIES

            runner_state, update_steps, instruct_sample = update_runner_state

            def _env_step(runner_state: RunnerState, _):
                train_state, env_state, last_obs, rng, update_i = (
                    runner_state.train_state,
                    runner_state.env_state,
                    runner_state.last_obs,
                    runner_state.rng,
                    runner_state.update_i,
                )

                if config.use_nlp and train_inst is not None:
                    last_obs = last_obs.replace(nlp_obs=instruct_sample.embedding)

                if config.vec_cont and train_inst is not None:
                    vmap_state_fn = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
                    cont_obs = vmap_state_fn(env_state.env_state.env_map, instruct_sample.condition, config.raw_obs)
                    last_obs = last_obs.replace(nlp_obs=cont_obs)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                # Squash the gpu dimension (network only takes one batch dimension)
                pi, value = network.apply(train_state.params, last_obs, rng=_rng)

                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.n_envs)

                # rng_step = rng_step.reshape((config.n_gpus, -1) + rng_step.shape[1:])
                vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
                # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))

                prev_env_state = env_state

                obsv, env_state, reward_env, done, info = vmap_step_fn(
                    rng_step, env_state, action, env_params
                )

                if train_inst is not None:
                    if train_inst is not None:
                        reward_batch = get_reward_batch(
                            instruct_sample.reward_i,
                            instruct_sample.condition,
                            prev_env_state.env_state.env_map,
                            env_state.env_state.env_map,
                            config.norm_reward
                        )
                else:
                    reward_batch = reward_env

                reward = jnp.where(done, reward_env, reward_batch)

                env_state = env_state.replace(
                    returned_episode_returns=(
                        env_state.returned_episode_returns - reward_env + reward
                    )
                )
                info["returned_episode_returns"] = env_state.returned_episode_returns

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = RunnerState(
                    train_state, env_state, obsv, rng, update_i=update_i
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = (
                runner_state.train_state,
                runner_state.env_state,
                runner_state.last_obs,
                runner_state.rng,
            )

            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.GAMMA * next_value * (1 - done) - value
                    gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):

                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        # Some reshaping to accomodate player, x, and y dimensions to action output. (Not used often...)
                        gae = gae[..., None, None, None]

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.CLIP_EPS,
                                1.0 + config.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.VF_COEF * value_loss
                            - config.ENT_COEF * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
                assert batch_size == config.num_steps * config.n_envs, (
                    "batch size must be equal to number of steps * number " + "of envs"
                )
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Save initial weight

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            metric = traj_batch.info

            rng = update_state[-1]

            # Save weight to checkpoint
            jax.debug.callback(
                save_checkpoint, runner_state, metric, steps_prev_complete
            )
            jax.debug.callback(_log_callback, metric)

            runner_state = RunnerState(
                train_state,
                env_state,
                last_obs,
                rng,
                update_i=runner_state.update_i + 1,
            )

            update_steps = update_steps + 1

            # Update instruction with config.instruct_freq using jax.lax.cond
            def _update_instruct():
                nonlocal rng
                random_indices = jax.random.randint(
                    rng, (config.n_envs,), 0, train_inst.reward_i.shape[0]
                )
                instruct_sample = jax.tree.map(lambda x: x[random_indices], train_inst)
                return instruct_sample

            if train_inst is not None:
                instruct_sample = jax.lax.cond(
                    update_steps % config.instruct_freq == 0,
                    lambda _: _update_instruct(),
                    lambda _: instruct_sample,  # No update if condition is false; keep train_inst as is
                    operand=None,
                )

            def _evaluate_step():

                nonlocal rng

                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config.n_envs)

                # sample n_envs rows from the instruct struct
                if test_inst is not None:
                    random_indices = jax.random.permutation(
                        rng,
                        jnp.arange(config.n_envs),
                    )[0 : config.n_envs]
                    instruct_sample = jax.tree.map(
                        lambda x: x[random_indices], test_inst
                    )
                else:
                    instruct_sample = jnp.zeros((config.n_envs, config.nlp_input_dim))

                def _env_step(carry, _):
                    rng, last_obs, state, done = carry

                    if config.use_nlp and test_inst is not None:
                        last_obs = last_obs.replace(nlp_obs=instruct_sample.embedding)

                    if config.vec_cont and test_inst is not None:
                        vmap_state_fn = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
                        cont_obs = vmap_state_fn(env_state.env_state.env_map, instruct_sample.condition, config.raw_obs)
                        last_obs = last_obs.replace(nlp_obs=cont_obs)

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    # Squash the gpu dimension (network only takes one batch dimension)

                    pi, value = network.apply(train_state.params, last_obs)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config.n_envs)

                    vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

                    obsv, next_state, reward_env, done, info = vmap_step_fn(
                        rng_step, state, action, env_params
                    )

                    if test_inst is not None:
                        reward_batch = get_reward_batch(
                            instruct_sample.reward_i,
                            instruct_sample.condition,
                            state.env_state.env_map,
                            next_state.env_state.env_map,
                            config.norm_reward
                        )
                    else:
                        reward_batch = reward_env

                    reward = jnp.where(done, reward_env, reward_batch)

                    next_state = next_state.replace(
                        returned_episode_returns=next_state.returned_episode_returns
                        - reward_env
                        + reward
                    )
                    info["returned_episode_returns"] = (
                        next_state.returned_episode_returns
                    )

                    transition = Transition(
                        done, action, value, reward, log_prob, obsv, info
                    )

                    return (rng, obsv, next_state, done), (transition, next_state)

                vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
                init_obs, init_state = vmap_reset_fn(
                    reset_rng, env_params, gen_dummy_queued_state(env)
                )
                done = jnp.zeros((config.n_envs,), dtype=bool)

                _, (traj_batch, states) = jax.lax.scan(
                    _env_step,
                    (rng, init_obs, init_state, done),
                    None,
                    length=int(
                        # config <- TrainConfig
                        # map_width: 16 -> 256
                        (config.map_width**2)
                        # max_board_scans: 3
                        * config.max_board_scans
                        # representation: "turtle" -> 2
                        * (2 if config.representation == "turtle" else 1)
                    ),
                )

                eval_metric = traj_batch.info

                states = jax.tree.map(
                    lambda x, y: jnp.concatenate([x[None], y], axis=0),
                    init_state,
                    states,
                )
                env0_state = jax.tree.map(lambda x: x[:, 0], states.env_state)

                frames = jax.vmap(env.render)(env0_state)

                _eval_callback = partial(
                    eval_callback,
                    config=config,
                    writer=multiple_handler,
                    train_start_time=train_start_time,
                    steps_prev_complete=steps_prev_complete,
                )

                jax.debug.callback(_eval_callback, eval_metric, metric, states, frames)

                if test_inst is not None:
                    loss = get_loss_batch(
                        reward_i=instruct_sample.reward_i,
                        condition=instruct_sample.condition,
                        env_maps=states.env_state.env_map[-2],
                    )

                    _loss_callback = partial(
                        loss_callback,
                        config=config,
                        writer=multiple_handler,
                    )

                    jax.debug.callback(_loss_callback, metric, loss)

                return None

            do_eval = (config.eval_freq != -1) and (
                update_steps % config.eval_freq == 0
            )
            _eval_step = _evaluate_step

            jax.lax.cond(
                do_eval,
                lambda _: _eval_step(),
                lambda _: None,
                operand=None,
            )

            return (runner_state, update_steps, instruct_sample), metric

        # Initialize the checkpoint at step 0
        jax.debug.callback(init_checkpoint, runner_state)

        _update_step = _update_step_with_render
        # Begin train

        # sample n_envs rows from the instruct struct
        if train_inst is not None:

            random_indices = jax.random.randint(
                runner_state.rng, (config.n_envs,), 0, train_inst.reward_i.shape[0]
            )
            instruct_sample = jax.tree.map(lambda x: x[random_indices], train_inst)

            logger.info(f"Instruction: {instruct_sample}")
        else:
            instruct_sample = None
            logger.info("Instruction: None")

        runner_state, metric = jax.lax.scan(
            _update_step,
            (runner_state, latest_update_step, instruct_sample),
            None,
            config.NUM_UPDATES - latest_update_step,
        )

        return {"runner_state": runner_state, "metrics": metric}

    return lambda rng: train(rng, config)


def init_checkpointer(config: Config) -> Tuple[Any, dict]:
    # This will not affect training, just for initializing dummy env etc. to load checkpoint.

    if config.encoder.separated_reward:
        config.encoder.output_dim = config.encoder.num_rewards * config.encoder.output_dim

    rng = jax.random.PRNGKey(30)
    # Set up checkpointing
    ckpt_dir = get_ckpt_dir(config)

    # Create a dummy checkpoint so we can restore it to the correct dataclasses
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    # env = FlattenObservationWrapper(env)

    env = LogWrapper(env)

    rng, _rng = jax.random.split(rng)
    network = init_network(env, env_params, config)
    init_x = env.gen_dummy_obs(env_params)
    # init_x = env.observation_space(env_params).sample(_rng)[None, ]
    network_params = network.init(_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config.MAX_GRAD_NORM),
        optax.adam(config.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)

    # reset_rng_r = reset_rng.reshape((config.n_gpus, -1) + reset_rng.shape[1:])
    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
    # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
    obsv, env_state = vmap_reset_fn(reset_rng, env_params, gen_dummy_queued_state(env))
    runner_state = RunnerState(
        train_state=train_state,
        env_state=env_state,
        last_obs=obsv,
        # ep_returns=jnp.full(config.num_envs, jnp.nan),
        rng=rng,
        update_i=0,
    )
    target = {"runner_state": runner_state, "step_i": 0}
    # Get absolute path
    ckpt_dir = os.path.abspath(ckpt_dir)

    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    # checkpoint_manager = orbax.checkpoint.CheckpointManager(
    #     ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), options)
    checkpoint_manager = ocp.CheckpointManager(
        # ocp.test_utils.erase_and_create_empty(ckpt_dir),
        ckpt_dir,
        options=options,
    )

    def try_load_ckpt(steps_prev_complete, target):
        runner_state = target["runner_state"]
        try:
            restored_ckpt = checkpoint_manager.restore(
                # steps_prev_complete, items=target)
                steps_prev_complete,
                args=ocp.args.StandardRestore(target),
            )
        except KeyError:
            # HACK
            runner_state = runner_state.replace(
                env_state=runner_state.env_state.replace(
                    env_state=runner_state.env_state.env_state.replace(
                        queued_state=gen_dummy_queued_state_old(env)
                    )
                )
            )
            target = {"runner_state": runner_state, "step_i": 0}
            restored_ckpt = checkpoint_manager.restore(
                steps_prev_complete, items=target
            )

        restored_ckpt["steps_prev_complete"] = steps_prev_complete
        if restored_ckpt is None:
            raise TypeError("Restored checkpoint is None")

        # HACK
        if isinstance(runner_state.env_state.env_state.queued_state, OldQueuedState):
            dummy_queued_state = gen_dummy_queued_state(env)

            # Now add leading dimension with sizeto match the shape of the original queued_state
            dummy_queued_state = jax.tree_map(
                lambda x: jnp.array(x, dtype=bool) if isinstance(x, bool) else x,
                dummy_queued_state,
            )
            dummy_queued_state = jax.tree_map(
                lambda x: jnp.repeat(x[None], config.n_envs, axis=0), dummy_queued_state
            )

            runner_state = restored_ckpt["runner_state"]
            runner_state = runner_state.replace(
                env_state=runner_state.env_state.replace(
                    env_state=runner_state.env_state.env_state.replace(
                        queued_state=dummy_queued_state,
                    )
                )
            )
            restored_ckpt["runner_state"] = runner_state

        return restored_ckpt

    if checkpoint_manager.latest_step() is None:
        restored_ckpt = None
    else:
        # print(f"Restoring checkpoint from {ckpt_dir}")
        # steps_prev_complete = checkpoint_manager.latest_step()

        ckpt_subdirs = os.listdir(ckpt_dir)
        ckpt_steps = [int(cs) for cs in ckpt_subdirs if cs.isdigit()]

        # Sort in decreasing order
        ckpt_steps.sort(reverse=True)
        for steps_prev_complete in ckpt_steps:
            try:
                restored_ckpt = try_load_ckpt(steps_prev_complete, target)
                if restored_ckpt is None:
                    raise TypeError("Restored checkpoint is None")
                break
            except TypeError as e:
                print(
                    f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}"
                )
                continue


    if config.encoder.ckpt_path is not None:
        logger.info(f"Restoring encoder checkpoint from {config.encoder.ckpt_path}")

        ckpt_subdirs = glob(join(config.encoder.ckpt_path, '*'))

        ckpt_steps = [int(basename(cs)) for cs in ckpt_subdirs if basename(cs).isdigit()]

        assert len(ckpt_steps) > 0, (
            "No checkpoint found in encoder checkpoint path. Set 'encoder.ckpt_path' to the paths ends with '**/ckpts'"
        )

        # Sort in decreasing order
        ckpt_steps.sort(reverse=True)
        for steps_prev_complete in ckpt_steps:
            ckpt_dir = os.path.join(config.encoder.ckpt_path, str(steps_prev_complete))

            try:
                from flax.training import checkpoints

                enc_state = checkpoints.restore_checkpoint(
                    ckpt_dir=ckpt_dir, target=None
                )
                assert enc_state is not None, "Restored params are None ({})".format(
                    ckpt_dir
                )

                enc_param = enc_state["params"]["params"]

                def get_encoder_params_recursive(params, key):
                    if key in params:
                        return params[key]
                    for v in params.values():
                        if isinstance(v, dict):
                            result = get_encoder_params_recursive(v, key)
                            if result is not None:
                                return result
                    return None

                enc_param = get_encoder_params_recursive(enc_param, "encoder")
                assert enc_param is not None, "Encoder not found in checkpoint"

                break

            except TypeError as e:
                logging.error(
                    f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}"
                )
                continue
    else:
        enc_param = None

    return checkpoint_manager, restored_ckpt, enc_param


def main_chunk(config, rng, exp_dir):
    """When jax jits the training loop, it pre-allocates an array with size equal to number of training steps. So, when training for a very long time, we sometimes need to break training up into multiple
    chunks to save on VRAM.
    """

    checkpoint_manager, restored_ckpt, encoder_param = init_checkpointer(config)

    if restored_ckpt is None:
        progress_csv_path = os.path.join(exp_dir, "progress.csv")
        assert not os.path.exists(progress_csv_path), (
            "Progress csv already exists, but have no checkpoint to restore "
            + "from. Run with `overwrite=True` to delete the progress csv."
        )
        # Create csv for logging progress
        with open(os.path.join(exp_dir, "progress.csv"), "w") as f:
            f.write("timestep,ep_return\n")

    train_jit = jax.jit(
        make_train(config, restored_ckpt, checkpoint_manager, encoder_param)
    )
    out = train_jit(rng)

    jax.block_until_ready(out)

    return out


@hydra.main(version_base=None, config_path="./conf", config_name="train_pcgrl")
def main(config: TrainConfig):

    if config.initialize is None or config.initialize:
        config = init_config(config)

    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir
    logger.info(f"running experiment at {exp_dir}")

    if config.wandb_key:
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f"{get_wandb_name(config)}-{dt}"
        wandb.login(key=config.wandb_key)
        wandb.init(
            project=config.wandb_project,
            group=get_group_name(config),
            entity=config.wandb_entity,
            name=get_wandb_name(config),
            id=wandb_id,
            save_code=True,
            config_exclude_keys=[
                "wandb_key",
                "_vid_dir",
                "_img_dir",
                "_numpy_dir",
                "overwrite",
                "initialize",
            ],
        )
        wandb.config.update(dict(config), allow_val_change=True)

    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    if config.timestep_chunk_size != -1:
        n_chunks = config.total_timesteps // config.timestep_chunk_size
        for i in range(n_chunks):
            config.total_timesteps = config.timestep_chunk_size + (
                i * config.timestep_chunk_size
            )
            print(f"Running chunk {i + 1}/{n_chunks}")
            out = main_chunk(config, rng, exp_dir)

    else:
        out = main_chunk(config, rng, exp_dir)


if __name__ == "__main__":
    main()
