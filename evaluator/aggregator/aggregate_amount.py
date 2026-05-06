import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Tiles


def aggregate_amount(
    env_map: chex.Array,
    tile_type: Dungeon3Tiles,
):
    return jnp.sum(env_map == tile_type, dtype=float)
