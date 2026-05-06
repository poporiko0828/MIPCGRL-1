import chex
import jax
import jax.numpy as jnp

from evaluator.aggregator import aggregate_direction


# Declare map size
rows, cols = 16, 16


def compute_direction_loss(
    env_map: chex.Array,
    tile_type: chex.Array,
    direction: chex.Array,
    rows: int = 16,
    cols: int = 16,
) -> chex.Array:
    """
    Args:
        env_map:
        tile_type:
        direction:
        rows (int, optional): map size (rows, 16 by default)
        cols (int, optional): map size (cols, 16 by default)
    """

    # vectorize direction
    direction = jnp.array(direction).flatten()

    tile_counts = aggregate_direction(env_map, tile_type, direction, rows, cols)
    opposite_tile_counts = aggregate_direction(
        env_map, tile_type, (direction + 2) % 4, rows, cols
    )

    loss = -tile_counts + opposite_tile_counts * 0.5 #  + penalty

    return loss
