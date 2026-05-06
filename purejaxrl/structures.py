import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
from typing import NamedTuple


class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_state: jnp.ndarray
    last_obs: jnp.ndarray
    rng: jnp.ndarray
    update_i: int


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray