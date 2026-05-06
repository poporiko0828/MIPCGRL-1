import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from ..losses import compute_region_loss


def evaluate_region(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
) -> chex.Array:

    prev_loss = jnp.abs(compute_region_loss(prev_env_map, cond, passable_tiles))
    curr_loss = jnp.abs(compute_region_loss(curr_env_map, cond, passable_tiles))

    reward = prev_loss - curr_loss
    reward = reward.astype(float)

    return reward
