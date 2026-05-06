import math
from datetime import datetime
from functools import partial
from typing import Tuple, Any
import imageio
import numpy as np
import optax
import pandas as pd
import os
import cv2
from os.path import basename, dirname, join, abspath
import hydra
import jax
import jax.numpy as jnp
import wandb
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
from tqdm import tqdm

from conf.config import EvalConfig, Config
from envs.pcgrl_env import gen_dummy_queued_state
from instruct_rl import NUM_CONDITIONS, NUM_FEATURES, FEATURE_NAMES
from instruct_rl.dataclass import Instruct
from instruct_rl.evaluate import get_loss_batch
from instruct_rl.evaluation.hamming import compute_hamming_distance
from instruct_rl.reward_set import get_reward_batch
from pcgrllm.utils.logger import get_wandb_name_eval
from purejaxrl.experimental.s5.wrappers import LogWrapper
from purejaxrl.structures import Transition, RunnerState
from train import init_checkpointer
from pcgrllm.utils.path_utils import (
    get_ckpt_dir,
    gymnax_pcgrl_make,
    init_config,
    init_network,
)

import logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))



def make_eval(config, restored_ckpt, encoder_params):
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env = LogWrapper(env)
    env.init_graphics()

    def eval(rng, runner_state):
        # INIT NETWORK
        network = init_network(env, env_params, config)

        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)

        if restored_ckpt is not None:
            network_params = restored_ckpt['runner_state'].train_state.params
        else:
            network_params = network.init(_rng, init_x)

        csv_path = abspath(join(dirname(__file__), "instruct", f"{config.eval_instruct_csv}.csv"))
        instruct_df = pd.read_csv(csv_path)
        # index to 'row_i'
        instruct_df = instruct_df.reset_index()
        instruct_df = instruct_df.rename(columns={'index': 'row_i'})

        instruct_df.to_csv(join(config.eval_dir, 'input.csv'), index=False)

        embedding_df = instruct_df.filter(regex="embed_*")
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



        condition_df = instruct_df.filter(regex="condition_*")
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

        instruct = Instruct(
            reward_i=reward_enum,
            condition=condition,
            embedding=embedding
        )

        if config.ANNEAL_LR:
            def linear_schedule(count):
                frac = (
                    1.0 - (count // (config.NUM_MINIBATCHES * config.update_epochs))
                    / config.NUM_UPDATES
                )
                return config.LR * frac
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # INIT ENV FOR TRAIN
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.n_envs)

        dummy_queued_state = gen_dummy_queued_state(env)

        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        obsv, env_state = vmap_reset_fn(reset_rng, env_params, dummy_queued_state)

        rng, _rng = jax.random.split(rng)

        if restored_ckpt is not None:
            runner_state = restored_ckpt["runner_state"]
        else:
            runner_state = RunnerState(train_state, env_state, obsv, rng, update_i=0)

        if encoder_params is not None:
            logger.info(f"Parameters loaded from encoder checkpoint ({config.encoder.ckpt_path})")
            runner_state.train_state.params['params']['subnet']['encoder'] = encoder_params

        # Expand instruct for n_eps
        n_envs = config.n_envs
        n_eps = config.n_eps

        eval_batches = sorted(np.tile(list(range(0, len(instruct_df), 1)), n_eps))
        eval_batches = jnp.array(eval_batches)

        n_rows = len(eval_batches)
        repetitions = np.tile(list(range(1, n_eps + 1, 1)), len(instruct_df))

        if len(eval_batches) != len(repetitions):
            raise Exception(f"Length of eval_batches and repetitions do not match {len(eval_batches)} != {len(repetitions)}")

        n_batches = math.ceil(n_rows / n_envs)

        losses, values, features = list(), list(), list()

        with tqdm(total=n_batches, desc="Evaluating Batches") as pbar:
            for batch_i in range(n_batches):
                # Get current batch
                start_idx = batch_i * n_envs
                end_idx = min((batch_i + 1) * n_envs, n_rows)
                idxes = eval_batches[start_idx:end_idx]
                batch_valid_size = len(idxes)

                batch_embedding = instruct.embedding[idxes]
                batch_condition = instruct.condition[idxes]
                batch_reward_i = instruct.reward_i[idxes]
                batch_repetition = repetitions[start_idx:end_idx]

                if len(batch_embedding) < n_envs:
                    batch_embedding = jnp.pad(batch_embedding,((0, n_envs - len(batch_embedding)), (0, 0)), mode="constant",)
                    batch_condition = jnp.pad(batch_condition,((0, n_envs - len(batch_condition)), (0, 0)), mode="constant",)
                    batch_reward_i = jnp.pad(batch_reward_i,((0, n_envs - len(batch_reward_i))), mode="constant",)
                    batch_repetition = jnp.pad(batch_repetition,((0, n_envs - len(batch_repetition))), mode="constant",)

                batch_instruct = Instruct(
                    reward_i=batch_reward_i,
                    condition=batch_condition,
                    embedding=batch_embedding
                )

                reset_rng = jnp.stack([jax.random.PRNGKey(seed) for seed in batch_repetition])

                init_obs, init_state = vmap_reset_fn(
                    reset_rng, env_params, gen_dummy_queued_state(env)
                )

                done = jnp.zeros((n_envs,), dtype=bool)

                @partial(jax.jit)
                def _env_step(carry, _):
                    rng, last_obs, state, done = carry

                    if config.use_nlp:
                        last_obs = last_obs.replace(nlp_obs=batch_instruct.embedding)

                    if config.vec_cont:
                        vmap_state_fn = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
                        cont_obs = vmap_state_fn(env_state.env_state.env_map, batch_instruct.condition, config.raw_obs)
                        last_obs = last_obs.replace(nlp_obs=cont_obs)

                    rng, _rng = jax.random.split(rng)

                    # SELECT ACTION
                    pi, value = network.apply(runner_state.train_state.params, last_obs)

                    action = pi.sample(seed=_rng)

                    log_prob = pi.log_prob(action)

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config.n_envs)

                    # STEP ENV
                    vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

                    obsv, next_state, reward_env, done, info = vmap_step_fn(
                        rng_step, state, action, env_params
                    )

                    reward_batch = get_reward_batch(
                        batch_instruct.reward_i,
                        batch_instruct.condition,
                        state.env_state.env_map,
                        next_state.env_state.env_map,
                        config.norm_reward
                    )
                    reward = jnp.where(done, reward_env, reward_batch)

                    next_state = next_state.replace(
                        returned_episode_returns=next_state.returned_episode_returns
                        - reward_env
                        + reward
                    )
                    info["returned_episode_returns"] = next_state.returned_episode_returns

                    transition = Transition(
                        done, action, value, reward, log_prob, obsv, info
                    )

                    return (rng, obsv, next_state, done), (transition, state)

                rng = jax.random.PRNGKey(30)

                _, (traj_batch, states) = jax.lax.scan(
                    _env_step,
                    (rng, init_obs, init_state, done),
                    None,
                    length=int(
                        (config.map_width**2)
                        * config.max_board_scans
                        * (2 if config.representation == "turtle" else 1)
                    ),
                )

                states = jax.tree.map(
                    lambda x, y: jnp.concatenate([x[None], y], axis=0),
                    init_state,
                    states,
                )

                @partial(jax.jit)
                def _env_render(env_state):
                    frames = jax.vmap(env.render)(env_state.env_state)
                    return frames

                last_states = jax.tree.map(lambda x: x[[-1], ...], states)
                rendered = jax.vmap(_env_render)(last_states) # (epi_length, n_envs, 288, 288, 4)

                # all_frames = jax.vmap(_env_render)(states) # (epi_length, n_envs, 288, 288, 4)
                # all_frames = all_frames.transpose(1, 0, 2, 3, 4)

                # Compute loss for the batch
                result = get_loss_batch(
                    reward_i=batch_instruct.reward_i,
                    condition=batch_instruct.condition,
                    env_maps=states.env_state.env_map[-1, :, :, :],
                )

                losses.append(result.loss)
                values.append(result.value)
                features.append(result.feature)

                rendered = rendered.transpose(1, 0, 2, 3, 4)
                rendered = np.array(rendered) # (n_envs, epi_length, 288, 288, 4)


                for idx, (row_i, reward_i, repeat_i, feature, state) in enumerate(zip(
                        idxes,
                        batch_reward_i[:batch_valid_size],
                        batch_repetition[:batch_valid_size],
                        result.feature[:batch_valid_size],
                        last_states.env_state.env_map[0, :][:batch_valid_size])
                ):
                    os.makedirs(f"{config.eval_dir}/reward_{row_i}/seed_{repeat_i}", exist_ok=True)

                    frames = rendered[idx]
                    # epi_frames = all_frames[idx]


                    for i, frame in enumerate(frames):


                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

                        frame = cv2.putText(
                            frame,
                            f"Region: {int(feature[0])} / Length: {int(feature[1])}",
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        imageio.imwrite(f"{config.eval_dir}/reward_{row_i}/seed_{repeat_i}/frame_{i}.png", frame)

                        np.save(f"{config.eval_dir}/reward_{row_i}/seed_{repeat_i}/state_{i}.npy", state)

                        # png_dir = os.path.join(config.eval_dir, f'reward_{row_i}/seed_{repeat_i}/png')
                        # os.makedirs(png_dir, exist_ok=True)
                        #
                        # for j, lvl_img in enumerate(epi_frames):
                        #     imageio.imwrite(f"{png_dir}/frame_{i}_{j}.png", lvl_img)
                        #
                        # imageio.mimsave(f"{config.eval_dir}/reward_{row_i}/seed_{repeat_i}/video_{i}.gif", epi_frames, duration=1/30, loop=1)

                        if wandb.run:
                            wandb.log({f"Image/reward_{row_i}/seed_{repeat_i}": wandb.Image(frame)})

                            if config.flush:
                                os.system(f"rm -r {config.eval_dir}/reward_{row_i}/seed_{repeat_i}/frame_{i}.png")

                pbar.update(1)


        losses = np.stack(losses, axis=0).reshape(-1)[:n_rows]
        # values = np.stack(values, axis=0).reshape(-1, NUM_CONDITIONS)[:n_rows]
        features = np.stack(features, axis=0).reshape(-1, NUM_FEATURES)[:n_rows]

        # get rows by index
        df_output = instruct_df.iloc[eval_batches]
        df_output = df_output.loc[:, ~df_output.columns.str.startswith("embed")]

        df_output['seed'] = repetitions

        # value_df = pd.DataFrame(values, columns=[f"value_{i}" for i in range(NUM_CONDITIONS)])
        # df_output = df_output.reset_index(drop=True)
        # value_df = value_df.reset_index(drop=True)
        # df_output = pd.concat([df_output, value_df], axis=1)

        df_output["loss"] = losses

        df_output = df_output.reset_index()

        # features for visualization
        feat_df = pd.DataFrame(features, columns=[f"feat_{i}" for i in FEATURE_NAMES]).reset_index()
        df_output = pd.concat([df_output, feat_df], axis=1)

        # save te eval
        df_output.to_csv(f"{config.eval_dir}/loss.csv", index=False)

        mean_loss = df_output.groupby('reward_enum').agg({'loss': ['mean']})
        mean_loss.columns = mean_loss.columns.droplevel(0)
        mean_loss = mean_loss.reset_index()

        dict_loss = dict()
        for _, row in mean_loss.iterrows():
            reward_enum, mean = row
            dict_loss[f'Loss/{str(int(reward_enum))}'] = mean

        if wandb.run:
            wandb.log(dict_loss)
            raw_table = wandb.Table(dataframe=df_output)
            wandb.log({'raw': raw_table})


        # Start of diversity evaluation
        scores = list()

        for row_i, row in tqdm(instruct_df.iterrows(), desc="Computing Diversity"):
            states = list()
            for seed_i in range(1, config.n_eps):
                state = np.load(f"{config.eval_dir}/reward_{row_i}/seed_{seed_i}/state_0.npy")
                states.append(state)
            states = np.array(states)
            score = compute_hamming_distance(states)
            scores.append(score)

        diversity_df = instruct_df.copy()
        diversity_df = diversity_df.loc[:, ~diversity_df.columns.str.startswith('embed')]
        diversity_df['diversity'] = scores

        if wandb.run:
            diversity_table = wandb.Table(dataframe=diversity_df)
            wandb.log({'diversity': diversity_table})

        if wandb.run and config.flush:
            for row_i, _ in instruct_df.iterrows():
                os.system(f"rm -r {config.eval_dir}/reward_{row_i}")

        return losses


    return lambda rng: eval(rng, config)


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
        rng=rng,
        update_i=0,
    )
    target = {"runner_state": runner_state, "step_i": 0}
    # Get absolute path
    ckpt_dir = os.path.abspath(ckpt_dir)

    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=False)
    # checkpoint_manager = orbax.checkpoint.CheckpointManager(
    #     ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), options)
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_dir,
        options=options,
    )

    def try_load_ckpt(steps_prev_complete, target):


        try:
            restored_ckpt = checkpoint_manager.restore(
                steps_prev_complete,
                args=ocp.args.StandardRestore(target, strict=False),
            )
        except Exception:
            restored_ckpt = checkpoint_manager.restore(
                steps_prev_complete,
                args=ocp.args.StandardRestore(target),

            )

        restored_ckpt["steps_prev_complete"] = steps_prev_complete
        if restored_ckpt is None:
            raise TypeError("Restored checkpoint is None")

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

        ckpt_subdirs = os.listdir(config.encoder.ckpt_path)
        ckpt_steps = [int(cs) for cs in ckpt_subdirs if cs.isdigit()]

        # Sort in decreasing order
        ckpt_steps.sort(reverse=True)
        for steps_prev_complete in ckpt_steps:

            ckpt_dir = os.path.join(config.encoder.ckpt_path, str(steps_prev_complete))

            try:
                from flax.training import checkpoints

                enc_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
                assert enc_state is not None, "Restored params are None ({})".format(ckpt_dir)

                enc_param = enc_state['params']['params']

                def get_encoder_params_recursive(params, key):
                    if key in params:
                        return params[key]
                    for v in params.values():
                        if isinstance(v, dict):
                            result = get_encoder_params_recursive(v, key)
                            if result is not None:
                                return result
                    return None

                enc_param = get_encoder_params_recursive(enc_param, 'encoder')
                assert enc_param is not None, "Encoder not found in checkpoint"

                break


            except TypeError as e:
                logging.error(f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}")
                continue
    else:
        enc_param = None


    return checkpoint_manager, restored_ckpt, enc_param

def main_chunk(config, rng):
    """When jax jits the training loop, it pre-allocates an array with size equal to number of training steps. So, when training for a very long time, we sometimes need to break training up into multiple
    chunks to save on VRAM.
    """

    if not config.random_agent:
        _, restored_ckpt, encoder_param = init_checkpointer(config)
    else:
        restored_ckpt, encoder_param = None, None

    eval_jit = make_eval(config, restored_ckpt, encoder_param)
    out = eval_jit(rng)
    jax.block_until_ready(out)

    return out



@hydra.main(version_base=None, config_path="./conf", config_name="eval_pcgrl")
def main(config: EvalConfig):
    config = init_config(config)

    if config.eval_aug_type is not None and config.eval_embed_type is not None and config.eval_instruct is not None:
        config.eval_instruct_csv = f'{config.eval_aug_type}/{config.eval_embed_type}/{config.eval_instruct}'

    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir
    logger.info(f"running experiment at {exp_dir}")

    eval_dir = os.path.join(exp_dir, f"eval_embed-{config.eval_embed_type}_inst-{config.eval_instruct}")
    config.eval_dir = eval_dir

    if config.reevaluate:
        if os.path.exists(eval_dir):
            logger.info(f"Removing existing evaluation directory at {eval_dir}")
            os.system(f"rm -r {eval_dir}")
        else:
            logger.info(f"No existing evaluation directory found at {eval_dir}")
    else:
        if os.path.exists(eval_dir):
            raise Exception(f"Evaluation directory already exists at {eval_dir}. Set reevaluate=True to overwrite.")

    os.makedirs(eval_dir, exist_ok=True)

    logger.info(f"running evaluation at {eval_dir}")

    if config.wandb_key:
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f"{get_wandb_name_eval(config)}-{dt}"
        wandb.login(key=config.wandb_key)
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=get_wandb_name_eval(config),
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


    main_chunk(config, rng)

if __name__ == '__main__':
    main()