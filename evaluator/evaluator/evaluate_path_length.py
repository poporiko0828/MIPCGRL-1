import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from ..losses import compute_path_length_loss, compute_region_loss


def evaluate_path_length(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
):


    prev_loss = jnp.abs(compute_path_length_loss(prev_env_map, cond, passable_tiles))
    curr_loss = jnp.abs(compute_path_length_loss(curr_env_map, cond, passable_tiles))

    prev_r_loss = jnp.abs(compute_region_loss(prev_env_map, 1, passable_tiles))
    curr_r_loss = jnp.abs(compute_region_loss(curr_env_map, 1, passable_tiles))

    reward = prev_loss - curr_loss
    reward += (prev_r_loss - curr_r_loss) * 0.5
    reward = reward.astype(float)

    reward = jnp.clip(reward, -5, 5)
    reward = jnp.sign(reward) * jnp.log1p(jnp.abs(reward))
    reward = jnp.clip(reward, -2, 2)

    return reward
