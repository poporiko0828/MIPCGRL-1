import json
import os
import logging
import gymnax
import jax
import numpy as np
import yaml
from os.path import basename, dirname

from conf.config import Config, EvoMapConfig, SweepConfig, TrainConfig, CollectConfig
from envs.candy import Candy, CandyParams
from envs.pcgrl_env import PROB_CLASSES, PCGRLEnvParams, PCGRLEnv, ProbEnum, RepEnum, get_prob_cls
from envs.play_pcgrl_env import PlayPCGRLEnv, PlayPCGRLEnvParams

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

DATASET_DIR = os.path.abspath(os.path.join(dirname(__file__), '..', "pcgrl_buffer"))


def get_exp_dir_evo_map(config: EvoMapConfig):
    exp_dir = os.path.join(
        'saves_evo_map',
        config.problem,
        f'pop-{config.evo_pop_size}_' +
        f'parents-{config.n_parents}_' +
        f'mut-{config.mut_rate}_' +
        f'{config.seed}_{config.exp_name}',
    )
    return exp_dir


def is_default_hiddims(config: Config):
    # Hack, because we're not consistent about when we truncate the hidden dims argument relative to getting the exp_dir
    # path.
    return tuple(config.hidden_dims) == (64, 256)[:len(config.hidden_dims)]


def get_dataset_dir(config):
    filename = (
        f"embed_type-{config.pretrained_model}_ft"
        if config.fine_tune
        else f"embed_type-{config.pretrained_model}"
    )
    filename = f"{filename}_inst-{config.instruct}"

    if not config.use_prev:
        filename = f"{filename}_prev-{str(config.use_prev).lower()}"

    if config.buffer_ratio < 1:
        filename = f"{filename}_bufratio-{config.buffer_ratio}"

    filename = f"{filename}.npz"

    filename = os.path.join(DATASET_DIR, filename)

    return filename


def get_exp_group(config):

    config_dict = {
        'inst': config.instruct
    }

    exp_group = os.path.join(
        '_'.join([f'{key}-{value}' for key, value in config_dict.items()])
    )

    flags_dict = {
        'fewshot': 'fs',
    }
    # Append suffixes for enabled flags
    for flag, suffix in flags_dict.items():
        if getattr(config, flag, False):  # Check if the flag exists and is True
            exp_group += f'_{suffix}'

    return exp_group


def get_short_target(target: str) -> str:
    # Split the target string into words
    words = target.split()

    # If there's only one word, return it with the length
    if len(words) == 1:
        return f"{words[0]}_{len(target)}"

    # Otherwise, take the first and last words and include the length
    return f"{words[0]}X{words[-1]}{len(target)}"


def get_exp_name(config):
    exp_group = get_exp_group(config)

    return f'{exp_group}'


def get_exp_dir(config):
    return os.path.abspath(os.path.join('saves', get_exp_name(config)))


def init_config(config: Config):
    config.n_gpus = jax.local_device_count()

    config.exp_group = get_exp_group(config)
    config.exp_dir = get_exp_dir(config)

    config.arf_size = (2 * config.map_width -
                       1 if config.arf_size == -1 else config.arf_size)

    config.vrf_size = (2 * config.map_width -
                       1 if config.vrf_size == -1 else config.vrf_size)

    if config.model == 'seqnca':
        config.hidden_dims = config.hidden_dims[:1]

    return config


def init_config_evo_map(config: EvoMapConfig):
    # FIXME: This is meaningless, should remove it eventually.
    config.arf_size = (2 * config.map_width -
                       1 if config.arf_size == -1 else config.arf_size)

    config.vrf_size = (2 * config.map_width -
                       1 if config.vrf_size == -1 else config.vrf_size)

    config.n_gpus = jax.local_device_count()
    config.exp_dir = get_exp_dir_evo_map(config)
    return config


def get_ckpt_dir(config: Config):
    return os.path.join(config.exp_dir, 'ckpts')


def get_env_params_from_config(config: Config):
    map_shape = ((config.map_width, config.map_width) if not config.is_3d
                 else (config.map_width, config.map_width, config.map_width))
    rf_size = max(config.arf_size, config.vrf_size)
    rf_shape = (rf_size, rf_size) if not config.is_3d else (rf_size, rf_size, rf_size)

    act_shape = tuple(config.act_shape)
    if config.is_3d:
        assert len(config.act_shape) == 3

    # Convert strings to enum ints
    problem = ProbEnum[config.problem.upper()]
    prob_cls = PROB_CLASSES[problem]
    ctrl_metrics = tuple([int(prob_cls.metrics_enum[c.upper()]) for c in config.ctrl_metrics])

    env_params = PCGRLEnvParams(
        problem=problem,
        representation=int(RepEnum[config.representation.upper()]),
        map_shape=map_shape,
        rf_shape=rf_shape,
        act_shape=act_shape,
        static_tile_prob=config.static_tile_prob,
        n_freezies=config.n_freezies,
        n_agents=config.n_agents,
        max_board_scans=config.max_board_scans,
        ctrl_metrics=ctrl_metrics,
        change_pct=config.change_pct,
        randomize_map_shape=config.randomize_map_shape,
        empty_start=config.empty_start,
        pinpoints=config.pinpoints,
        nlp_input_dim=config.nlp_input_dim if config.use_nlp else -1,
    )
    return env_params


def get_play_env_params_from_config(config: Config):
    map_shape = (config.map_width, config.map_width)
    rf_size = max(config.arf_size, config.vrf_size)
    rf_shape = (rf_size, rf_size) if not config.is_3d else (rf_size, rf_size, rf_size)

    return PlayPCGRLEnvParams(
        map_shape=map_shape,
        rf_shape=rf_shape,
    )


def gymnax_pcgrl_make(env_name, config: Config, **env_kwargs):
    if env_name in gymnax.registered_envs:
        return gymnax.make(env_name)

    elif env_name == 'PCGRL':
        env_params = get_env_params_from_config(config)
        env = PCGRLEnv(env_params)

    elif env_name == 'PlayPCGRL':
        env_params = get_play_env_params_from_config(config)
        env = PlayPCGRLEnv(env_params)

    elif env_name == 'Candy':
        env_params = CandyParams()
        env = Candy(env_params)

    return env, env_params


def get_sweep_conf_path(cfg: SweepConfig):
    conf_sweeps_dir = os.path.join('conf', 'sweeps')
    # sweep_conf_path_json = os.path.join(conf_sweeps_dir, f'{cfg.name}.json')
    sweep_conf_path_yaml = os.path.join(conf_sweeps_dir, f'{cfg.name}.yaml')
    return sweep_conf_path_yaml


def write_sweep_confs(_hypers: dict, eval_hypers: dict):
    conf_sweeps_dir = os.path.join('conf', 'sweeps')
    os.makedirs(conf_sweeps_dir, exist_ok=True)
    for grid_hypers in _hypers:
        name = grid_hypers['NAME']
        save_grid_hypers = grid_hypers.copy()
        save_grid_hypers['eval_hypers'] = eval_hypers
        with open(os.path.join(conf_sweeps_dir, f'{name}.yaml'), 'w') as f:
            f.write(yaml.dump(save_grid_hypers))
        # with open(os.path.join(conf_sweeps_dir, f'{name}.json'), 'w') as f:
        #     f.write(json.dumps(grid_hypers, indent=4))


def load_sweep_hypers(cfg: SweepConfig):
    sweep_conf_path = get_sweep_conf_path(cfg)
    if os.path.exists(sweep_conf_path):
        hypers = yaml.load(open(sweep_conf_path), Loader=yaml.FullLoader)
        eval_hypers = hypers.pop('eval_hypers')
    else:
        raise FileNotFoundError(f"Could not find sweep config file {sweep_conf_path}")
    return hypers, eval_hypers
