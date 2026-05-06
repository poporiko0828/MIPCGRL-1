import jax
import jax.numpy as jnp
import chex
from jax import jit


def regional_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    tile_type: int,
    region: int
) -> float:
    """
    Calculate rewards for placing or removing tiles in correct/incorrect regions of the map.

    Args:
        prev_env_map (chex.Array): Previous environment map (16x16).
        curr_env_map (chex.Array): Current environment map (16x16).
        tile_type (int): Tile type to evaluate.
        region (int): Target region

    Returns:
        float: Total reward for tile placements or removals.
    """
    # Split map into 3x3 regions (each region is ~5x5 tiles for a 16x16 map)
    n_rows, n_cols = curr_env_map.shape
    row_step, col_step = n_rows // 3, n_cols // 3

    # Create masks for each region as boolean arrays
    def north_mask():
        mask = jnp.zeros((n_rows, n_cols), dtype=bool)
        mask = mask.at[:row_step, col_step:col_step * 2 + 1].set(True)
        return mask

    def east_mask():
        mask = jnp.zeros((n_rows, n_cols), dtype=bool)
        mask = mask.at[row_step: 2 * row_step + 1:, 2 * col_step:].set(True)
        return mask

    def south_mask():
        mask = jnp.zeros((n_rows, n_cols), dtype=bool)
        mask = mask.at[2 * row_step:, col_step: 2 * col_step + 1].set(True)
        return mask

    def west_mask():
        mask = jnp.zeros((n_rows, n_cols), dtype=bool)
        mask = mask.at[row_step: 2 * row_step + 1, :col_step].set(True)
        return mask


    # Use jax.lax.switch to select the correct mask
    masks = [north_mask, east_mask, south_mask, west_mask]
    correct_mask = jax.lax.switch(region, masks)

    print(correct_mask.astype(int))

    # Extract the region masks for current and previous maps
    prev_correct_region = prev_env_map[correct_mask] == tile_type
    curr_correct_region = curr_env_map[correct_mask] == tile_type

    # Compute rewards for correct placements
    correct_additions = jnp.sum(curr_correct_region & ~prev_correct_region)
    correct_reward = correct_additions * 1.0  # +1 for correctly placed tiles

    # Compute penalties for incorrect placements
    incorrect_mask_slices = [
        south_mask,
    ]
    total_reward = correct_reward
    return total_reward

prev_map = jnp.array([
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

curr_map = jnp.array([
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

reward_west = regional_reward(prev_map, curr_map, tile_type=2, region=3)

print("Reward (West):", reward_west)
# print("Reward (East):", reward_east)
