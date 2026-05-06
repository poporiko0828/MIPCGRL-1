import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible
from instruct_rl.dataclass import NormalizationWeights

from ..losses import compute_region_loss


def evaluate_region_llm(
    curr_env_map: chex.Array,
    cond: chex.Array,
    weights: chex.Array = NormalizationWeights,
    passable_tiles: chex.Array = Dungeon3Passible,
) -> chex.Array:

    curr_loss = compute_region_loss(curr_env_map, cond, passable_tiles)

    reward = curr_loss.astype(float)

    # normalize the region reward
    reward = jnp.divide(reward, weights)

    return reward
