import os
import logging
import gymnax
import jax
from glob import glob
import yaml
from os.path import basename, abspath, join
import re

from encoder.model import apply_encoder_model
from conf.config import Config, EvoMapConfig, SweepConfig, TrainConfig, EncoderConfig
from envs.candy import Candy, CandyParams
from envs.pcgrl_env import PROB_CLASSES, PCGRLEnvParams, PCGRLEnv, ProbEnum, RepEnum
from envs.play_pcgrl_env import PlayPCGRLEnv, PlayPCGRLEnvParams
from models import ActorCritic, ActorCriticPCGRL, AutoEncoder, ConvForward, ConvForward2, Dense, \
    NCA, SeqNCA, NLPConvForward, EncoderNLPConvForward

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


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


def get_exp_group(config):
    if config.env_name == 'PCGRL':

        if config.use_nlp or config.vec_cont:
            nlp_dict = {
                'embed': config.embed_type,
                'inst': config.instruct,
            }

            if config.encoder.model:
                nlp_dict['enc'] = config.encoder.model
                nlp_dict['tr'] = str(config.encoder.trainable).lower()[0]
        else:
            nlp_dict = {}

        # task
        config_dict = {
            'model': config.model,
            'exp': config.exp_name,
        }

        enc_def_setting = EncoderConfig()
        tr_def_setting = TrainConfig()

        # RQ4 parameters
        if config.buffer_ratio != tr_def_setting.buffer_ratio:
            config_dict['br'] = config.buffer_ratio
        if config.encoder.output_dim != enc_def_setting.output_dim:
            config_dict['es'] = config.encoder.output_dim

        if hasattr(config, 'random_agent') and config.random_agent:
            config_dict['model'] = 'rand'

        config_dict = {**nlp_dict, **config_dict}


        exp_group = os.path.join(
            '_'.join([f'{key}-{value}' for key, value in config_dict.items()])
        )

        if config.encoder.use_cls:
            exp_group = exp_group + '_cls'
        if config.encoder.separated_reward:
            exp_group = exp_group + '_st'
        if config.encoder.use_mask:
            exp_group = exp_group + '_mk'

        flags_dict = {
            'vec_cont': 'vec',
            'raw_obs': 'ro',
        }
        # Append suffixes for enabled flags

        for flag, suffix in flags_dict.items():
            if getattr(config, flag, False):  # Check if the flag exists and is True
                exp_group += f'_{suffix}'

    elif config.env_name == 'PlayPCGRL':
        exp_group = os.path.join(
            'saves',
            f'play_w-{config.map_width}_' + \
            f'{config.model}-{config.activation}_' + \
            f'vrf-{config.vrf_size}_arf-{config.arf_size}_' + \
            f'{config.exp_name}'
        )
    elif config.env_name == 'Candy':
        exp_group = os.path.join(
            'candy_' + \
            f'{config.exp_name}'
        )
    else:
        exp_group = os.path.join(
            config.env_name
        )
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

    # target_character = get_short_target(config.target_character) if config.task == 'scenario' else config.target_character

    return f'{exp_group}_s-{config.seed}'


def get_exp_dir(config):
    return os.path.join('saves', get_exp_name(config))


def get_encoder_ckpt_dir(config, exp_dirs):
    encoder_flags = {
        "cls": config.encoder.use_cls,
        "st": config.encoder.separated_reward,
        "mk": config.encoder.use_mask
    }

    for key, is_enabled in encoder_flags.items():
        pattern = rf"(?:^|_)({key})(?:_|$)"

        if is_enabled:
            exp_dirs = [d for d in exp_dirs if re.search(pattern, d)]
        else:
            exp_dirs = [d for d in exp_dirs if not re.search(pattern, d)]

    return exp_dirs



def init_config(config: Config):
    config.n_gpus = jax.local_device_count()

    if config.aug_type is not None and config.embed_type is not None and config.instruct is not None:
        config.instruct_csv = f'{config.aug_type}/{config.embed_type}/{config.instruct}'

    if config.instruct_csv is not None:
        if hasattr(config, 'vec_cont') and config.vec_cont is True:
            config.use_nlp = False
            config.vec_input_dim = 9
            config.nlp_input_dim = 0
        else:
            config.use_nlp = True

            if config.model == 'conv':
                config.model = 'nlpconv'
                logger.info("Setting model to `nlpconv` due to the instruct set")


    if config.vec_cont is True and config.model != 'contconv':
        config.model = 'contconv'
        logger.warning("Setting model to `contconv` due to the vec_cont flag")

    if config.encoder.model is not None:
        logger.info(f'Loading checkpoint for the encoder model: {config.encoder.model} '
                    f'(embed size: {config.encoder.output_dim}, buffer_ratio: {config.buffer_ratio})')
        try:
            ckpt_dir = abspath(config.encoder.ckpt_dir)

            exp_dirs = glob(join(ckpt_dir, '*'))

            logger.info(f"exp_dirs: {exp_dirs}")
            embed_type_keyword = f'enc-{config.encoder.model}_'
            embed_size_keyword = f'es-{config.encoder.output_dim}_'
            buffer_ratio_keyword = f'br-{config.buffer_ratio}_'

            exp_dirs = [
                d for d in exp_dirs
                if embed_type_keyword in d and embed_size_keyword in d and buffer_ratio_keyword in d
            ]

            exp_dirs = get_encoder_ckpt_dir(config, exp_dirs)
            logger.info(f"exp_dirs: {exp_dirs}")

            if len(exp_dirs) == 0:
                raise FileNotFoundError(f"Could not find encoder checkpoint for {config.encoder.model} "
                                        f"with embed size {config.encoder.output_dim} and buffer ratio {config.buffer_ratio}")
            elif len(exp_dirs) > 1:
                raise FileExistsError(f"Multiple encoder checkpoints found for {config.encoder.model} "
                                        f"with embed size {config.encoder.output_dim} and buffer ratio {config.buffer_ratio}")

            config.encoder.ckpt_path = join(exp_dirs[0], 'ckpts')

            logger.info(f"Encoder checkpoint set to [{config.encoder.ckpt_path}]")
        except Exception as e:
            logger.error(f"Error loading encoder checkpoint: {e}")
            exit(-1)

    if config.representation in set({'wide', 'nca'}):
        # TODO: Technically, maybe arf/vrf size should affect kernel widths in (we're assuming here) the NCA model?
        config.arf_size = config.vrf_size = config.map_width

    if config.representation == 'nca':
        config.act_shape = (config.map_width, config.map_width)

    else:
        config.arf_size = (2 * config.map_width -
                           1 if config.arf_size == -1 else config.arf_size)

        config.vrf_size = (2 * config.map_width -
                           1 if config.vrf_size == -1 else config.vrf_size)

    if hasattr(config, 'evo_pop_size') and hasattr(config, 'n_envs'):
        assert config.n_envs % (config.evo_pop_size * 2) == 0, "n_envs must be divisible by evo_pop_size * 2"
    if config.model == 'conv2':
        config.arf_size = config.vrf_size = min([config.arf_size, config.vrf_size])

    config.exp_group = get_exp_group(config)
    config.exp_dir = get_exp_dir(config)

    config._vid_dir = os.path.join(config.exp_dir, 'videos')
    config._img_dir = os.path.join(config.exp_dir, 'images')
    config._numpy_dir = os.path.join(config.exp_dir, 'numpy')

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


def init_network(env: PCGRLEnv, env_params: PCGRLEnvParams, config: Config):
    if config.env_name == 'Candy':
        # In the candy-player environment, action space is flat discrete space over all candy-direction combos.
        action_dim = env.action_space(env_params).n

    elif 'PCGRL' in config.env_name:
        action_dim = env.rep.action_space.n
        # First consider number of possible tiles
        # action_dim = env.action_space(env_params).n
        # action_dim = env.rep.per_tile_action_dim

    else:
        action_dim = env.num_actions


    if config.vec_cont is True and config.model != 'contconv':
        logger.warning("Setting model to `contconv` due to the vec_cont flag")
        config.model = 'contconv'

    if config.model == "dense":
        network = Dense(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, vrf_size=config.vrf_size,
        )

    elif config.model == "nlpconv" or config.model == 'contconv':

        network = EncoderNLPConvForward(
            config=config.encoder,
            encoder=apply_encoder_model(config.encoder, config.encoder.dropout_rate, config.encoder.broadcast_dropout) if config.encoder.model else None,
            train_encoder=config.encoder.trainable,
            nlp_conv_forward=NLPConvForward(
                action_dim=action_dim, activation=config.activation,
                arf_size=config.arf_size, act_shape=config.act_shape,
                vrf_size=config.vrf_size,
                nlp_input_dim=config.nlp_input_dim,
                hidden_dims=config.hidden_dims,
                separated_reward=config.encoder.separated_reward,
            )
        )

    elif config.model == "conv":
        network = ConvForward(
            action_dim=action_dim, activation=config.activation,
            arf_size=config.arf_size, act_shape=config.act_shape,
            vrf_size=config.vrf_size,
            hidden_dims=config.hidden_dims,
        )

    elif config.model == "conv2":
        network = ConvForward2(
            action_dim=action_dim, activation=config.activation,
            act_shape=config.act_shape,
            hidden_dims=config.hidden_dims,
        )
    elif config.model == "seqnca":
        network = SeqNCA(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, act_shape=config.act_shape,
            vrf_size=config.vrf_size,
            hidden_dims=config.hidden_dims,
        )
    elif config.model in {"nca", "autoencoder"}:
        if config.model == "nca":
            network = NCA(
                representation=config.representation,
                tile_action_dim=env.rep.tile_action_dim,
                activation=config.activation,
            )
        elif config.model == "autoencoder":
            network = AutoEncoder(
                representation=config.representation,
                action_dim=action_dim,
                activation=config.activation,
            )
    else:
        raise Exception(f"Unknown model {config.model}")
    # if config.env_name == 'PCGRL':
    if 'PCGRL' in config.env_name:
        network = ActorCriticPCGRL(network, act_shape=config.act_shape,
                                   n_agents=config.n_agents, n_ctrl_metrics=len(config.ctrl_metrics),
                                   nlp_input_dim=env_params.nlp_input_dim, model_type=config.model)
    # elif config.env_name == 'PlayPCGRL':
    #     network = ActorCriticPlayPCGRL(network)
    else:
        network = ActorCritic(network)
    return network


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
        nlp_input_dim=config.nlp_input_dim,
        vec_input_dim=config.vec_input_dim,
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
