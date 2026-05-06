import chex
import jax.numpy as jnp

from ..losses import compute_direction_loss
from envs.probs.dungeon3 import Dungeon3Passible
from instruct_rl.dataclass import NormalizationWeights


# TODO: I want to write a reward logic that gives a stronger penalty the longer there is no improvement.
# Currently, the evaluation code can only consider the previous state.
# It may be necessary to consider historical data for the evaluation metric.
# This method will be important when the direction metric starts to improve and considers the number of entities.


def evaluate_direction_llm(
    curr_env_map: chex.Array,
    cond: chex.Array,
    weights: chex.Array = NormalizationWeights,
    tile_type: chex.Array = Dungeon3Passible,
    rows: int = 16,
    cols: int = 16,
):
    """
    Aggregates the number of entities summoned in the specified direction and calculates how much the ability to generate as per the user's intent has improved.

    Args:
        curr_env_map (chex.Array): Current map state
        cond (chex.Array): Desired direction by the user
        tile_type (Dungeon3Tiles): Tile type to aggregate
        rows (int, optional): map size (rows, 16 by default)
        cols (int, optional): map size (cols, 16 by default)
    """

    curr_loss = compute_direction_loss(curr_env_map, tile_type, cond, rows, cols)

    reward = curr_loss.astype(float)

    # normalize the direction reward
    reward = jnp.divide(reward, weights)

    return reward
