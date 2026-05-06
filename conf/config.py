from typing import Iterable, List, Optional, Tuple, Union
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

@dataclass
class Config:
    lr: float = 1.0e-4
    n_envs: int = 4
    # How many steps do I take in all of my batched environments before doing a gradient update
    num_steps: int = 128
    total_timesteps: int = int(5e7)
    timestep_chunk_size: int = -1
    update_epochs: int = 10
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    activation: str = "relu"
    env_name: str = "PCGRL"
    ANNEAL_LR: bool = False
    DEBUG: bool = True
    exp_name: str = "def"
    seed: int = 0

    problem: str = "dungeon3"
    representation: str = "turtle"
    model: str = "nlpconv"
    norm_reward: float = 0.15

    # NLP params
    use_nlp: bool = False
    nlp_input_dim: int = 768

    vec_cont: bool = False
    vec_input_dim: Optional[int] = None
    raw_obs: bool = False

    map_width: int = 16
    randomize_map_shape: bool = False
    is_3d: bool = False
    # ctrl_metrics: Tuple[str] = ('diameter', 'n_regions')
    ctrl_metrics: Tuple[str, ...] = ()
    # Size of the receptive field to be fed to the action subnetwork.
    vrf_size: Optional[
        int
    ] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16
    # Size of the receptive field to be fed to the value subnetwork.
    arf_size: Optional[
        int
    ] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16
    # TODO: actually take arf and vrf into account in models, where possible

    change_pct: float = -1.0

    # The shape of the (patch of) edit(s) to be made by the edited by the generator at each step.
    act_shape: Tuple[int, int] = (1, 1)

    static_tile_prob: Optional[float] = 0.0
    n_freezies: int = 0
    n_agents: int = 1  # multi-agent is fake and broken
    multiagent: bool = False
    max_board_scans: float = 3.0

    # How many milliseconds to wait between frames of the rendered gifs
    gif_frame_duration: int = 25

    """ DO NOT USE. WILL BE OVERWRITTEN. """
    exp_dir: Optional[str] = None
    n_gpus: int = 1

    # use hpc
    use_hpc: bool = False

    # use prev state
    use_prev: bool = True

    # normalize reward
    normal_weigth: float = 5

    # To make the task simpler, always start with an empty map
    empty_start: bool = False

    # In problems with tile-types with specified valid numbers, fix/freeze their random placement at the beginning of
    # each episode.
    pinpoints: bool = False

    hidden_dims: Tuple[int, ...] = (64, 256)

    # TODO: Implement this. Just a placeholder for now.
    reward_every: int = 1

    # A toggle, will add `n_envs` to the experiment name if we are profiling training FPS, so that we can distinguish
    # results.
    profile_fps: bool = False

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    initialize: Optional[bool] = None

    # Wandb
    wandb_key: Optional[str] = None

    wandb_project: Optional[str] = 'multi_instruct_pcgrl'
    wandb_entity: Optional[str] = None
    wandb_resume: str = 'allow'
    evaluator: str = 'hr'  # 'vit', 'hr' (heuristic)

    exp_group: Optional[str] = None

    _vid_dir: Optional[str] = None
    _img_dir: Optional[str] = None
    _numpy_dir: Optional[str] = None

    aug_type: Optional[str] = 'test'
    embed_type: Optional[str] = 'bert'
    instruct: Optional[str] = None

    instruct_csv: Optional[str] = None


@dataclass
class EncoderConfig(Config):
    model: Optional[str] = None

    deterministic: bool = True
    num_layers: int = 2  # 1 ~ 3
    hidden_dim: int = 256
    output_dim: int = 64
    num_heads: int = 8  # 2, 4, 8, 16, 32, 64 etc
    freeze: bool = False
    buffer_ratio: float = 1

    ckpt_dir: str = "./encoder_weights"
    ckpt: Optional[str] = None
    dropout_rate: float = 0.0
    broadcast_dropout: bool = False  # Use a broadcasted dropout along batch dims.

    # DO NOT SET THIS
    ckpt_path: Optional[str] = None

    trainable: bool = False

    use_cls: bool = True

    separated_reward: bool = True
    num_rewards: int = 5
    use_mask: bool = True

@dataclass
class DecoderConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 1
    dropout_rate: float = 0.0


@dataclass
class EvoMapConfig(Config):
    n_generations: int = 100_000
    evo_pop_size: int = 100
    n_parents: int = 50
    mut_rate: float = 0.3
    render_freq: int = 10_000
    log_freq: int = 1_000
    callbacks: bool = True


@dataclass
class TrainConfig(Config):
    overwrite: bool = False

    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(5e6)

    # Render after this many update steps
    render_freq: int = 50
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 5
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None
    ###########################################################################

    agents: int = 1
    current_iteration: int = -1

    instruct_freq: int = 1
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    buffer_ratio: float = 1


@dataclass
class DebugConfig(Config):
    overwrite: bool = True

    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(1)
    # Render after this many update steps
    render_freq: int = 1000
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 1
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None
    ###########################################################################

    total_timesteps: int = int(1e6)
    log_freq: int = 1


@dataclass
class TrainAccelConfig(TrainConfig):
    evo_freq: int = 10
    evo_pop_size: int = 10
    evo_mutate_prob: float = 0.1

    instruct_freq: int = 1


@dataclass
class EvalConfig(TrainConfig):
    reevaluate: bool = False

    random_agent: bool = False
    # In how many bins to divide up each metric being evaluated
    n_bins: int = 10
    n_eval_envs: int = 10
    n_eps: int = 10
    eval_map_width: Optional[int] = None
    eval_max_board_scans: Optional[int] = None
    eval_randomize_map_shape: Optional[bool] = None
    eval_seed: int = 0

    eval_aug_type: str = "test"
    eval_embed_type: str = "bert"
    eval_instruct: str = "whole"
    eval_instruct_csv: Optional[str] = None
    eval_dir: Optional[str] = None
    eval_map_types: int = 5

    wandb_project: str = 'eval_pcgrl'

    metrics_to_keep: Tuple[str] = ("mean_ep_reward",)
    flush: bool = True


@dataclass
class SweepConfig(EvalConfig, TrainConfig):
    wandb_project: Optional[str] = 'sweep_pcgrl'



@dataclass
class EnjoyConfig(EvalConfig):
    random_agent: bool = False
    # How many episodes to render as gifs
    n_eps: int = 5
    eval_map_width: Optional[int] = None
    render_stats: bool = True
    n_enjoy_envs: int = 1
    render_ims: bool = False


@dataclass
class ProfileEnvConfig(Config):
    N_PROFILE_STEPS: int = 5000
    reevaluate: bool = False


@dataclass
class SweepConfig(EvalConfig, TrainConfig):
    name: Optional[str] = None
    mode: str = "train"
    slurm: bool = True


@dataclass
class TrainLLMConfig(Config):
    overwrite: bool = False

    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(5e6)
    # Render after this many update steps
    total_timesteps: int = int(5e7)

    render_freq: int = 40
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 100
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # Prompt

    num_scenario: int = 1
    start_scenario: int = 1
    end_scenario: int = 2  # 1, 2, 3, 6, 9

    train_ratio: float = 0.8
    prompt_path: str = "instruct/scenario_prompt.json"

    # Augmentation
    typo_noise: bool = False
    similar_words: bool = False
    permutation: bool = False

    # Hyper parameters
    pretrained_model: str = "bert"  # bert, roberta, albert, electra
    model_size: str = "base"  # small, base, large

    epochs: int = 100
    max_length: int = 128  # 32: num_scenario < 6, 64: 6 <= num_scenario < 10
    batch_size: int = 32
    instruct_csv: Optional[str] = None
    offline: bool = True

    # Encoder hyper parameters

    hidden_dims: int = 512

    num_layers: int = 1  # 1 ~ 3
    output_dim: int = 512
    num_heads: int = 8  # 2, 4, 8, 16, 32, 64 etc

    encoding_type: str = 'mlp'  # mlp, sa (self-attention), mp(mean pool), mlp_vae

    deterministic: bool = True
    num_samples: int = 100

    # decoder parameters
    decoder_hidden_dims: int = 512
    decoder_num_layers: int = 1
    decoder_output_dim: int = 1
    buffer_path: Optional[str] = "pcgrl_buffer/test_data.npz"

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None
    ###########################################################################

    # Eval rollout setting
    random_agent: bool = False
    eval_map_width: Optional[int] = 16
    eval_max_board_scans: Optional[int] = 3
    eval_randomize_map_shape: Optional[bool] = False
    eval_seed: int = 0
    n_eval_envs: int = 1
    reevaluate: bool = True
    n_eps: int = 2

    # validation setting
    reward_function_path: Optional[str] = None

    # reward generation setting
    bypass_reward_path: Optional[str] = None
    bypass_train_path: Optional[str] = None

    evaluator: str = "vit"  # 'vit', 'hr' (heuristic)

    n_samples: int = 30

    n_codegen_trials: int = 3
    n_codefix_trials: int = 3

@dataclass
class CollectConfig(TrainConfig):
    wandb_project: str = 'collect_traj'


    traj_path: Optional[str] = None
    traj_freq: int = 2  # Update basis
    traj_max_envs: int = 3
    traj_step_freq: int = 1  # Step basis

    total_timesteps: int = int(2e7)

    aug_type: str = "test"
    embed_type: str = "test"
    model: str = "conv"

    nlp_input_dim: int = 0


@dataclass
class BertConfig(Config):

    overwrite: bool = True

    num_samples: int = 100
    batch_size: int = 32
    offline: bool = True
    pretrained_model: str = 'bert'
    model_size: str = "base"
    buffer_path: str = "./pcgrl_buffer"
    fine_tune: bool = False

    deterministic: bool = True
    hidden_dims: int = 512
    num_layers: int = 1  # 1 ~ 3
    output_dim: int = 512
    num_heads: int = 8  # 2, 4, 8, 16, 32, 64 etc

    # decoder parameters
    decoder_hidden_dims: int = 512
    decoder_num_layers: int = 1
    decoder_output_dim: int = 1

    buffer_ratio: float = 1
    instruct: str = "whole"


@dataclass
class BertTrainConfig(BertConfig):
    wandb_project: str = 'embedding'

    max_length: int = 128

    batch_size: int = 512
    lr: float = 0.001
    n_epochs: int = 100
    n_buffer: int = -1
    use_prev: bool = False

    encoder: EncoderConfig = field(default_factory=EncoderConfig)

@dataclass
class BertEvalConfig(BertConfig):
    wandb_project: str = 'eval_bert'
    use_prev: bool = False
    buffer_ratio: float = 1

    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class RewardConfig(Config):
    dir_prefix: str = "encoder-"
    overwrite: bool = True
    n_max_points: int = 1000
    embed_visualize_freq: int = 5

    num_samples: int = 100
    batch_size: int = 32

    num_layers: int = 2  # 1 ~ 3
    hidden_dim: int = 512
    output_dim: int = 1

    figure_dir: str = "figures"
    buffer_dir: str = "./pcgrl_buffer"
    n_buffer: int = -1
    train_ratio: float = 0.8
    n_epochs: int = 100
    separated_reward: bool = False

    weight_decay: float = 1e-4
    normal_weigth: float = 5

    augment: bool = True
    zero_reward_ratio: Optional[float] = None
    buffer_ratio: float = 1

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    deterministic: bool = True
    use_prev: bool = False

    steps_per_epoch: Optional[int] = None
    warmup_epochs: int = 10  # set 10% of the total timesteps

@dataclass
class RewardTrainConfig(RewardConfig):
    wandb_project: str = 'train_reward'

    n_envs: int = 300
    ckpt_freq: int = 5


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="evo_map_pcgrl", node=EvoMapConfig)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="debug_pcgrl", node=DebugConfig)
cs.store(name="train_accel_pcgrl", node=TrainAccelConfig)
cs.store(name="enjoy_pcgrl", node=EnjoyConfig)
cs.store(name="eval_pcgrl", node=EvalConfig)
cs.store(name="profile_pcgrl", node=ProfileEnvConfig)
cs.store(name="sweep_pcgrl", node=SweepConfig)
cs.store(name="collect_pcgrl", node=CollectConfig)

# PCGRLLM Configs
cs.store(name="train_pcgrllm", node=TrainLLMConfig)

cs.store(name="train_bert", node=BertTrainConfig)
cs.store(name="eval_bert", node=BertEvalConfig)

cs.store(name="train_reward", node=RewardTrainConfig)

