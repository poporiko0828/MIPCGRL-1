import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from ..aggregator import aggregate_region


def compute_region_loss(
    env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
) -> chex.Array:
    n_regions = aggregate_region(env_map, passable_tiles).astype(float)

    loss = jnp.subtract(n_regions, cond)

    return loss
