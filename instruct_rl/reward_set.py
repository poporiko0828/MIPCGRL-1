import jax
from jax import vmap
import chex
import jax.numpy as jnp
from os.path import join, abspath, dirname

from functools import partial
from envs.pathfinding import FloodRegions, FloodPath, calc_diameter
from envs.probs.dungeon3 import Dungeon3Tiles, Dungeon3Passible
from evaluator import (
    evaluate_direction,
    evaluate_amount,
    evaluate_region,
    evaluate_path_length,
)


@partial(jax.jit, static_argnames=("norm_reward",))
def get_reward_batch(
    reward_i: chex.Array,
    condition: chex.Array,
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    norm_reward: float = 1.0,
) -> chex.Array:
    """
    Compute batch rewards by mapping indices to reward functions and executing them in parallel.

    Args:
        reward_i: Array of indices mapping to functions in call_reward.
        condition: Array of conditions corresponding to each reward calculation.
        prev_env_map: Previous environment map.
        curr_env_map: Current environment map.

    Returns:
        rewards: Array of computed rewards.
    """
    # List of reward functions
    reward_funcs = [
        lambda cond, prev_map, curr_map: 0.0,  # 0
        lambda cond, prev_map, curr_map: evaluate_region(
            prev_map, curr_map, cond[0]
        ),  # 1 (region)
        lambda cond, prev_map, curr_map: evaluate_path_length(
            prev_map, curr_map, cond[1]
        ),  # 2 (diameter)
        lambda cond, prev_map, curr_map: evaluate_amount(
            prev_map, curr_map, cond[2], Dungeon3Tiles.WALL.value, norm_reward
        ),  # 3 (block)
        lambda cond, prev_map, curr_map: evaluate_amount(
            prev_map, curr_map, cond[3], Dungeon3Tiles.BAT.value, norm_reward
        ),  # 4 (bat_amount)
        lambda cond, prev_map, curr_map: evaluate_direction(
            prev_map, curr_map, cond[4], Dungeon3Tiles.BAT.value, norm_reward
        ),  # 5 (bat_direction)
        lambda cond, prev_map, curr_map: 0.0,
    ]

    # Map indices to functions using `switch`
    def compute_reward(func_idx, cond_value, _prev_env_map, _curr_env_map):
        reward_values = vmap(lambda idx: jax.lax.switch(idx, reward_funcs, cond_value, _prev_env_map, _curr_env_map))(func_idx)
        return jnp.sum(reward_values)


    compute_reward_vmap = vmap(compute_reward, in_axes=(0, 0, 0, 0))
    rewards = compute_reward_vmap(reward_i, condition, prev_env_map, curr_env_map)

    return jax.lax.stop_gradient(rewards)


if __name__ == "__main__":
    import pandas as pd

    instruct_csv = abspath(
        join(dirname(__file__), "..", "instruct", "scenario_prompt2.csv")
    )

    instruct_df = pd.read_csv(instruct_csv)
    embedding_df = instruct_df.filter(regex="embed_*")
    embedding_df = embedding_df.reindex(
        sorted(embedding_df.columns, key=lambda x: int(x.split("_")[-1])), axis=1
    )
    embedding = embedding_df.to_numpy()
    embedding = jnp.pad(
        embedding, ((0, 0), (0, 716 - embedding.shape[1])), mode="constant"
    )

    from instruct_rl.dataclass import Instruct

    df_cond = instruct_df.filter(regex='condition_*')
    condition_df = df_cond.reindex(sorted(df_cond.columns, key=lambda x: int(x.split('_')[-1])), axis=1)
    condition = condition_df.to_numpy()

    print(instruct_df["reward_enum"].to_list())
    reward_enum_list = [[int(digit) for digit in str(num)] for num in instruct_df["reward_enum"].to_list()]
    max_len = max(len(x) for x in reward_enum_list)
    split_result = jnp.array([
        x + [0] * (max_len - len(x)) for x in reward_enum_list
    ])
    print(split_result)

    instruct = Instruct(
        reward_i=split_result,
        condition=jnp.array(condition),
        embedding=embedding,
    )

    # Dummy environment maps
    prev_env_map = jnp.zeros((9, 16, 16))
    curr_env_map = jnp.ones((9, 16, 16))

    # Compute rewards
    rewards = get_reward_batch(
        instruct.reward_i, instruct.condition, prev_env_map, curr_env_map
    )
