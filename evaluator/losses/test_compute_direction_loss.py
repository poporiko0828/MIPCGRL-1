import jax
import chex
import jax.numpy as jnp


from envs.probs.dungeon3 import Dungeon3Tiles

from .compute_direction_loss import compute_direction_loss
from ..direction_type import Direction


rows, cols = 16, 16
tile = Dungeon3Tiles.BAT.value
direction = Direction.west.value


def test_compute_direction_loss():
    global rows, cols, n_tile

    # sampling map
    env_maps: chex.Array = jnp.zeros((4, rows, cols))
    env_maps = env_maps.at[0, 0, 0].set(tile)  # left-top
    env_maps = env_maps.at[1, rows - 1, 0].set(tile)  # right-top
    env_maps = env_maps.at[2, 0, cols - 1].set(tile)  # left-bottom
    env_maps = env_maps.at[3, rows - 1, cols - 1].set(tile)  # right-bottom

    losses = jax.lax.map(
        lambda env_map: compute_direction_loss(env_map, tile, direction),
        env_maps,
    )

    print(env_maps)
    print(losses)


if __name__ == "__main__":
    test_compute_direction_loss()
