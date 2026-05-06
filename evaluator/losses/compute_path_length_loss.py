import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from ..aggregator import aggregate_path_length


def compute_path_length_loss(
    env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
):
    path_length = aggregate_path_length(env_map, passable_tiles).astype(float)

    loss = jnp.subtract(path_length, cond)

    return loss
