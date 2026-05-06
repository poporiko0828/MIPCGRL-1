import chex
import jax.numpy as jnp
import jax

from functools import partial

from envs.probs.dungeon3 import Dungeon3Tiles
from instruct_rl.dataclass import NormalizationWeights


def evaluate_amount_llm(
    curr_env_map: chex.Array,
    cond: chex.Array,
    weights: chex.Array = NormalizationWeights,
    tile_type: chex.Array = Dungeon3Tiles,
) -> chex.Array:

    curr_amount = jnp.sum(curr_env_map == tile_type)

    curr_loss = jnp.subtract(curr_amount, cond)

    reward = curr_loss
    reward = reward.astype(float)

    # normalize the amount reward
    reward = jnp.divide(reward, weights)

    return reward

