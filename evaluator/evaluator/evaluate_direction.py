import chex
import jax

from functools import partial
from envs.probs.dungeon3 import Dungeon3Tiles
from evaluator.losses import compute_direction_loss


# TODO: I want to write a reward logic that gives a stronger penalty the longer there is no improvement.
# Currently, the evaluation code can only consider the previous state.
# It may be necessary to consider historical data for the evaluation metric.
# This method will be important when the direction metric starts to improve and considers the number of entities.


@partial(jax.jit, static_argnames=("norm_weight",))
def evaluate_direction(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_type: chex.Array,
    norm_weight: float = 1.0,
    rows: int = 16,
    cols: int = 16,
):
    """
    Aggregates the number of entities summoned in the specified direction and calculates how much the ability to generate as per the user's intent has improved.

    Args:
        prev_env_map (chex.Array): Previous map state
        curr_env_map (chex.Array): Current map state
        cond (chex.Array): Desired direction by the user
        tile_type (Dungeon3Tiles): Tile type to aggregate
        rows (int, optional): map size (rows, 16 by default)
        cols (int, optional): map size (cols, 16 by default)
    """

    prev_loss = compute_direction_loss(prev_env_map, tile_type, cond, rows, cols)
    curr_loss = compute_direction_loss(curr_env_map, tile_type, cond, rows, cols)

    reward: chex.Array = prev_loss - curr_loss
    reward = reward.astype(float)
    reward = reward.clip(-2, 2) * norm_weight

    return reward
