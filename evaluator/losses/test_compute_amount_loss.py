import jax
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Tiles

from .compute_amount_loss import compute_amount_loss


rows, cols = 16, 16
n_tile = 10


def test_compute_amonunt_loss():
    global rows, cols, n_tile

    # sampling map

    rng = jax.random.PRNGKey(42)
    col_rng, row_rng = jax.random.split(rng, 2)

    indices_col, indices_row = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols))
    indices_col, indices_row = indices_col.flatten(), indices_row.flatten()
    indices_col = jax.random.permutation(col_rng, indices_col)[:n_tile]
    indices_row = jax.random.permutation(row_rng, indices_row)[:n_tile]

    tile = Dungeon3Tiles.BAT.value

    env_map = jnp.zeros((16, 16))
    for row, col in zip(indices_row, indices_col):
        env_map = env_map.at[row, col].set(tile)

    print(env_map)

    loss = compute_amount_loss(env_map, tile, 0)
    print(loss)


if __name__ == "__main__":
    test_compute_amonunt_loss()
