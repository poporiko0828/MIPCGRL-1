import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Tiles

from ..aggregator import aggregate_amount


def compute_amount_loss(
    env_map: chex.Array,
    tile_type: Dungeon3Tiles,
    cond: chex.Array,
) -> chex.Array:

    loss = jnp.abs(jnp.subtract(aggregate_amount(env_map, tile_type), cond)).astype(
        float
    )

    return loss
