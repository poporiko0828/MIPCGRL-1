import chex
import jax
import jax.numpy as jnp

from typing import List, Callable

from .evaluators.region import evaluate_region
from .evaluators.path_length import evaluate_path_length
from .evaluate_amount import evaluate_amount
from .evaluate_direction import evaluate_direction

from envs.probs.dungeon3 import Dungeon3Tiles

EvaluatorSet: List[Callable[[chex.Array, chex.Array, chex.Array], chex.Array]] = [
    lambda prev_map, curr_map, cond: 0.0,
    lambda prev_map, curr_map, cond: evaluate_region(prev_map, curr_map, cond).astype(
        float
    ),  # 1 (region)
    lambda prev_map, curr_map, cond: evaluate_path_length(
        prev_map, curr_map, cond
    ),  # 2 (path_length)
    lambda prev_map, curr_map, cond: evaluate_amount(
        prev_map, curr_map, cond, Dungeon3Tiles.WALL.value
    ),  # 3 (block)
    lambda prev_map, curr_map, cond: evaluate_amount(
        prev_map, curr_map, cond, Dungeon3Tiles.BAT.value
    ),  # 4 (bat_amount)
    lambda prev_map, curr_map, cond: evaluate_amount(
        prev_map, curr_map, cond, Dungeon3Tiles.SCORPION.value
    ),  # 5 (scorpion_amount)
    lambda prev_map, curr_map, cond: 0.0,  # 6
    lambda prev_map, curr_map, cond: 0.0,  # 7
    # lambda prev_map, curr_map, cond: evaluate_direction(
    #     prev_map, curr_map, cond, Dungeon3Tiles.BAT.value
    # ),  # 6 (bat_direction)
    # lambda prev_map, curr_map, cond: evaluate_direction(
    #     prev_map, curr_map, cond, Dungeon3Tiles.SCORPION.value
    # ),  # 7 (scorpion_direction)
]


@jax.jit
def evaluate_combinational(
    prev_map: chex.Array,
    curr_map: chex.Array,
    cond: chex.Array,
    combinations: chex.Array,
    weights: chex.Array,
) -> chex.Array:

    def compute_reward(func_idx, prev_map, curr_map, cond):
        return jax.lax.switch(
            func_idx,
            EvaluatorSet,
            prev_map,
            curr_map,
            cond,
        )

    rewards = jax.vmap(compute_reward, in_axes=(0, None, None, 0), out_axes=0)(
        combinations,
        prev_map,
        curr_map,
        cond,
    )

    weighted_rewards = rewards * weights

    return jnp.sum(weighted_rewards, dtype=float)
