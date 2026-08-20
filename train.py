import argparse
from datetime import datetime
import logging
import os
from glob import glob
from omegaconf import OmegaConf  # 💡 これをここに追加！

import threading
import sys
import select
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
from os.path import abspath, join, dirname
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


# --- 既存の get_train_test と同等の処理を外出しにするヘルパー ---
def load_new_instruction_from_csv(csv_name, config, max_instructs=128):
    """新しいCSVから指示を読み込み、固定サイズ(max_instructs)にパディングして返すヘルパー"""
    import os
    import pandas as pd
    import jax.numpy as jnp
    import numpy as np
    
    # パスを解決 (環境に合わせて調整してください。以下は一例です)
    # 確実に存在する絶対パスを指定
    csv_path = f"/home/hosozawa/MIPCGRL_local/MIPCGRL/instruct/{csv_name}.csv"

    if not os.path.exists(csv_path):
        print(f" エラー: 指定されたファイルが見つかりません: {csv_path}")
        return None
            
    try:
        df = pd.read_csv(csv_path)
        actual_rows = df.shape[0]
        if actual_rows > max_instructs:
            print(f"警告: CSVの行数({actual_rows})が最大サイズ({max_instructs})を超えています。切り捨てます。")
            df = df.iloc[:max_instructs]
            actual_rows = max_instructs

        # 各種データの抽出
        reward_i = df["reward_enum"].to_numpy()[:, None]
        cond_cols = [c for c in df.columns if c.startswith("condition_")]
        condition = df[cond_cols].to_numpy()
        embed_cols = [c for c in df.columns if c.startswith("embed_")]
        embedding = df[embed_cols].to_numpy()

        # 固定サイズバッファの作成
        padded_reward = np.zeros((max_instructs, reward_i.shape[1]), dtype=np.int32)
        padded_cond = np.zeros((max_instructs, condition.shape[1]), dtype=np.int32) # 型をint32に統一
        padded_embed = np.zeros((max_instructs, embedding.shape[1]), dtype=np.float32)
        padded_mask = np.zeros((max_instructs,), dtype=bool)

        # データの流し込み
        padded_reward[:actual_rows] = reward_i
        padded_cond[:actual_rows] = condition
        padded_embed[:actual_rows] = embedding
        padded_mask[:actual_rows] = True

        return {
            "reward_i": padded_reward,
            "condition": padded_cond,
            "embedding": padded_embed,
            "mask": padded_mask
        }
    except Exception as e:
        print(f" CSVの読み込み中にエラーが発生しました: {e}")
        return None


def check_pause_callback(update_i, current_reward_i, current_condition, current_embedding, current_mask, nlp_input_dim):
    """JAXの内部からは、50回に1回ログを出すだけの軽量な処理に変更"""
    if int(update_i) % 50 == 0:
        print(f"\n[ Current Gym Instruct @ Update {update_i} ]")
        print(f" -> reward_enum (パースされた報酬ID): {current_reward_i[0].tolist() if hasattr(current_reward_i, 'tolist') else current_reward_i[0]}")
        print(f" -> condition (条件ベクトル): {current_condition[0].tolist() if hasattr(current_condition, 'tolist') else current_condition[0]}")
        print("-" * 50)
    
    # 💡 常に現在の指示をそのままJAX側に返します（ここでは入力待ちをしない）
    return (
        current_reward_i, 
        current_condition, 
        current_embedding,
        current_mask
    )




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

        # 💡 【追加項目】もし外側のループから累積ステップが渡されていれば、それを足す
        if hasattr(config, "cumulative_step_offset"):
            t = config.cumulative_step_offset + t

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

            instruct_df = pd.read_csv(csv_path, sep=None, engine='python')

            def get_train_test(df, is_train=True):
                # ★ 修正: train列の表記揺れ（文字列の "True", "TRUE", "1" など）を安全にブール値に変換
                if "train" in df.columns:
                    # 文字列や大文字小文字を統一して比較
                    train_mask = df["train"].astype(str).str.upper().str.strip()
                    if is_train:
                        df = df[train_mask.isin(["TRUE", "1", "1.0"])]
                    else:
                        df = df[train_mask.isin(["FALSE", "0", "0.0"])]
                else:
                    raise KeyError(f"CSVファイルに 'train' 列が見つかりません。列名を確認してください。カラム一覧: {list(df.columns)}")

                # データが0行になっていないかチェック
                if len(df) == 0:
                    mode_str = "train == True" if is_train else "train == False"
                    raise ValueError(f"指示CSV【{config.instruct_csv}】から {mode_str} となるデータが1行も検出されませんでした。ファイルの中身やフォーマットを確認してください。")

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

                # ★ 修正: インデックスエラーを避けるため、絞り込んだ後の df から抽出
                reward_enum_list = [[int(digit) for digit in str(num)] for num in df["reward_enum"].to_list()]

                max_len = max(len(x) for x in reward_enum_list) if reward_enum_list else 1

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


            

            # ========================================================
            # ★ アイデア1対応: マスクを用いた固定サイズ動的サンプリング（アプローチA適用）
            # ========================================================
            # 初回のみプール変数を初期化
            if 'current_pool_reward_i' not in locals():
                max_instructs = 128
                actual_rows = train_inst.reward_i.shape[0]
                
                # 元のデータの型(train_inst.***.dtype)をそのまま引き継ぐ
                padded_reward = jnp.zeros((max_instructs, train_inst.reward_i.shape[1]), dtype=train_inst.reward_i.dtype)
                padded_cond = jnp.zeros((max_instructs, train_inst.condition.shape[1]), dtype=train_inst.condition.dtype)
                padded_embed = jnp.zeros((max_instructs, train_inst.embedding.shape[1]), dtype=train_inst.embedding.dtype)
                
                # 初期プールを流し込む
                current_pool_reward_i = padded_reward.at[:actual_rows].set(train_inst.reward_i)
                current_pool_condition = padded_cond.at[:actual_rows].set(train_inst.condition)
                current_pool_embedding = padded_embed.at[:actual_rows].set(train_inst.embedding)
                
                # 初期マスクの生成 (有効行だけTrue)
                padded_mask = jnp.zeros((max_instructs,), dtype=jnp.bool_)
                current_pool_mask = padded_mask.at[:actual_rows].set(True)

            # JAX側への結果形状の通知（128行固定）
            result_shapes = (
                jax.ShapeDtypeStruct(current_pool_reward_i.shape, current_pool_reward_i.dtype),
                jax.ShapeDtypeStruct(current_pool_condition.shape, current_pool_condition.dtype),
                jax.ShapeDtypeStruct(current_pool_embedding.shape, current_pool_embedding.dtype),
                jax.ShapeDtypeStruct(current_pool_mask.shape, current_pool_mask.dtype),
            )

            # --------------------------------------------------------
            # 💡 【アプローチA】指定したステップ頻度の時だけ CPU 問い合わせを実行
            # --------------------------------------------------------
            def _fetch_latest_pool(_):
                return jax.pure_callback(
                    check_pause_callback,
                    result_shapes,
                    update_steps,
                    current_pool_reward_i,
                    current_pool_condition,
                    current_pool_embedding,
                    current_pool_mask,
                    config.nlp_input_dim,
                )

            def _keep_current_pool(_):
                return (
                    current_pool_reward_i,
                    current_pool_condition,
                    current_pool_embedding,
                    current_pool_mask,
                )

            # update_steps % config.instruct_freq == 0 の時だけ pure_callback を発火
            new_pool_reward_i, new_pool_condition, new_pool_embedding, new_pool_mask = jax.lax.cond(
                update_steps % config.instruct_freq == 0,
                _fetch_latest_pool,
                _keep_current_pool,
                operand=None,
            )
            
            current_pool_reward_i = new_pool_reward_i
            current_pool_condition = new_pool_condition
            current_pool_embedding = new_pool_embedding
            current_pool_mask = new_pool_mask
            # --------------------------------------------------------

            # 有効なインデックスから安全にサンプリングする関数
            def _update_instruct():
                nonlocal rng
                raw_indices = jax.random.randint(
                    rng, (config.n_envs,), 0, current_pool_reward_i.shape[0]
                )
                
                valid_indices = jnp.where(current_pool_mask, size=current_pool_mask.shape[0], fill_value=0)[0]
                num_valid = jnp.maximum(1, jnp.sum(current_pool_mask))
                
                safe_indices = valid_indices[jnp.mod(raw_indices, num_valid)]
                
                return instruct_sample.replace(
                    reward_i=current_pool_reward_i[safe_indices],
                    condition=current_pool_condition[safe_indices],
                    embedding=current_pool_embedding[safe_indices]
                )

            if train_inst is not None:
                instruct_sample = jax.lax.cond(
                    update_steps % config.instruct_freq == 0,
                    lambda _: _update_instruct(),
                    lambda _: instruct_sample,
                    operand=None,
                )

            #==================================================================================

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


def main_chunk(config, exp_dir, rng):
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    network = init_network(env, env_params, config)
    checkpoint_manager, restored_ckpt, encoder_params = init_checkpointer(config)

    # 💡 1チャンクあたりのステップ数をシステム本来の「30,720歩」に固定
    INTERVAL_STEPS = 30720
    original_total_timesteps = config.total_timesteps
    

    # JAX側に「1回あたり30,720歩」を総ステップ数として誤認させ、評価タイミングと同期させる
    config.total_timesteps = INTERVAL_STEPS  

    train_init = make_train(config, restored_ckpt, checkpoint_manager, encoder_params)
    train_jit = jax.jit(train_init)

    n_chunks = original_total_timesteps // INTERVAL_STEPS
    if n_chunks == 0:
        n_chunks = 1

    logger.info(f"学習を開始します (全 {n_chunks} チャンク / 1チャンク={INTERVAL_STEPS} steps)")
    print("\n🚀 【高速・メモリ節約モード】ノンストップで自動連投を実行します。")
    print("💡 指示を変更したい場合は、学習中にターミナルで 'p' を入力してEnterを押しておいてください。")
    print("   次の節目のタイミング（30,720歩ごと）で自動的に検知して一時停止します。\n")

    import sys
    import select

    cumulative_step = 0  # 💡 対策②: ループ回数による進捗の迷子を防ぐ累積カウンター
    out = None
    
    for chunk_i in range(n_chunks):
        logger.info(f"=== チャンク {chunk_i + 1} / {n_chunks} を実行中 (累積開始ステップ: {cumulative_step}) ===")
        
        train_start_time = timer()
        # ========================================================
        # 📊 【ステップ1】jax.profiler の仕込み
        # 1チャンク目はウォームアップ（JITコンパイル）として流し、
        # 2チャンク目（chunk_i == 1）の時だけ データを取得する
        # ========================================================
        should_profile = (chunk_i == 1)  # 2チャンク目でプロファイル実行

        if should_profile:
            trace_dir = os.path.join(exp_dir, "tensorboard_trace")
            print(f"\n[📊 PROFILER] プロファイリングを開始します（5秒後に自動停止します）... (保存先: {trace_dir})")
            
            # トレース開始
            jax.profiler.start_trace(trace_dir)
            
            # ⏱️ 別スレッドで5秒後に stop_trace() を呼ぶタイマーを仕込む
            def stop_profiler():
                try:
                    jax.profiler.stop_trace()
                    print("\n[📊 PROFILER] 5秒経過したため、プロファイリングを自動停止・保存しました！\n")
                except Exception as e:
                    pass  # すでに停止している場合などの安全策

            timer_thread = threading.Timer(5.0, stop_profiler)
            timer_thread.start()

        # 1. GPU処理を実行
        out = train_jit(rng)
        out = jax.block_until_ready(out) # 完了を待つ

        # 万が一5秒未満で終わった場合のクリーンアップ
        if should_profile:
            timer_thread.cancel()  # タイマー解除
            try:
                jax.profiler.stop_trace()
            except Exception:
                pass
        # ========================================================
        
      
        
        # 📸 2. 【今回追加するリネーム処理】
        # JAXが保存した固定名（30720）のファイルを、累積ステップ数の名前に変更して避難させる
        try:
            import shutil
            
            # 今回の本当の合計ステップ数（例: 30720, 61440, 92160...）
            actual_step = cumulative_step + INTERVAL_STEPS
            
            # 元々の保存先パスを再現
            base_img_path = os.path.join(config._img_dir, "image_30720.png")
            base_vid_path = os.path.join(config._vid_dir, "video_30720.gif")
            
            # 新しい連番の保存先パス
            new_img_path = os.path.join(config._img_dir, f"image_{actual_step}.png")
            new_vid_path = os.path.join(config._vid_dir, f"video_{actual_step}.gif")
            
            # ファイルが存在すれば、上書きされる前に別名でコピー（または移動）
            if os.path.exists(base_img_path):
                shutil.copyfile(base_img_path, new_img_path)
                logger.info(f"🔄 連番画像を生成しました: image_{actual_step}.png")
                
            if os.path.exists(base_vid_path):
                shutil.copyfile(base_vid_path, new_vid_path)
                logger.info(f"🔄 連番GIFを生成しました: video_{actual_step}.gif")
                
        except Exception as rename_err:
            logger.warning(f"⚠️ 連番ファイルへの変更中にスキップ可能なエラーが発生しました: {rename_err}")

        # 3. 今回の周回分（30,720歩）を累積に加算して次の周へ
        cumulative_step += INTERVAL_STEPS
        
        # --- この後に既存のキーボードチェック（select.select） ---
        
        # 乱数状態（rng）を次の周回へ安全に引き継ぐ
        try:
            if isinstance(out, tuple) and len(out) > 0:
                runner_state = out[0]
                if hasattr(runner_state, "rng"):
                    rng = runner_state.rng
                elif isinstance(runner_state, dict) and "rng" in runner_state:
                    rng = runner_state["rng"]
        except Exception as e:
            logger.warning(f"rng の引き継ぎに失敗しました（現在の rng をキープします）: {e}")

        # 💡 非同期キーボードチェック (0.001秒だけ標準入力を覗き見する)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
        
        if rlist:
            user_input = sys.stdin.readline().strip()
            # ユーザーが事前に 'p' + Enter を入力していた場合のみ介入
            if user_input == 'p':
                print(f"\n[⏸️ ユーザー要求により 累積 {cumulative_step} steps 時点で一時停止しました]")
                print("="*50)
                print("新しい指示CSVファイル名を入力してください（例: manybats）")
                print("="*50)
                new_csv = input("New instruct_csv >> ").strip()
                if new_csv:
                    new_data = load_new_instruction_from_csv(new_csv, config, max_instructs=128)
                    if new_data is not None:
                        config.instruct_csv = new_csv
                        print(f"✅ 成功: 指示を【{new_csv}】に更新しました。")

                        # ==========================================================
                        # 🌟 【ここを追加】指示変更ログの自動保存処理
                        # ==========================================================
                        try:
                            log_path = os.path.join(exp_dir, "instruction_log.csv")
                            
                            # 記録するデータの準備
                            # ※ total_stepsが計算できるタイミングの現在のステップ数（直近のstep）
                            current_step = cumulative_step
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # 新規作成か追記かを判定してデータフレームを作成
                            log_df = pd.DataFrame([{
                                "timestamp": timestamp,
                                "total_steps": current_step,
                                "instruct_csv": config.instruct_csv
                            }])
                            
                            # ファイルが存在しない場合はヘッダー付き、存在する場合はヘッダーなしで追記
                            if not os.path.exists(log_path):
                                log_df.to_csv(log_path, index=False)
                            else:
                                log_df.to_csv(log_path, mode='a', header=False, index=False)
                                
                            print(f"💾 指示変更ログを保存しました: {log_path}")
                        except Exception as log_err:
                            print(f"⚠️ ログ保存中にエラーが発生しました: {log_err}")
                        # ==========================================================
                        
                        print("JAX関数を新しい指示で再コンパイルしています... (少し時間がかかります)")
                        train_init = make_train(config, restored_ckpt, checkpoint_manager, encoder_params)
                        train_jit = jax.jit(train_init)
                    else:
                        print("❌ 読み込み失敗。指示は変更されませんでした。")
                print("▶️ 学習を再開します。変更したい場合は再度 'p' + Enter を入力してください。\n")

    # 最後に一応元の設定に戻しておく
    config.total_timesteps = original_total_timesteps
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

    # 古いチェックポイントの削除
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    # 呼び出し側も正しい引数の順番 (config, exp_dir, rng) に統一
    out = main_chunk(config, exp_dir, rng)


if __name__ == "__main__":
    main()