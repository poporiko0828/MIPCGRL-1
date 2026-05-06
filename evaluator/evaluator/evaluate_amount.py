import chex
import jax.numpy as jnp
import jax

from functools import partial
from envs.probs.dungeon3 import Dungeon3Tiles


@partial(jax.jit, static_argnames=("norm_weight",))
def evaluate_amount(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_type: Dungeon3Tiles,
    norm_weight: float = 1.0,
) -> chex.Array:

    prev_amount = jnp.sum(prev_env_map == tile_type)
    curr_amount = jnp.sum(curr_env_map == tile_type)

    prev_loss = jnp.abs(jnp.subtract(prev_amount, cond))
    curr_loss = jnp.abs(jnp.subtract(curr_amount, cond))

    reward = prev_loss - curr_loss
    reward = reward.astype(float)
    reward = reward * norm_weight

    return reward
