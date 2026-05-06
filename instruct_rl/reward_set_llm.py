import jax
from jax import vmap
import chex
import jax.numpy as jnp
from os.path import join, abspath, dirname


from envs.probs.dungeon3 import Dungeon3Tiles
from instruct_rl.dataclass import NormalizationWeights
from evaluator import (
    evaluate_direction_llm,
    evaluate_amount_llm,
    evaluate_region_llm,
    evaluate_path_length_llm,
)


@jax.jit
def get_reward_batch_llm(
    reward_i: chex.Array,
    condition: chex.Array,
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    normal_weights: chex.Array,
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
        lambda cond, curr_map: 0.0,  # 0
        lambda cond, curr_map: evaluate_region_llm(
            curr_map, cond[0], jnp.divide(NormalizationWeights.REGION.value, normal_weights)
        ),  # 1 (region)
        lambda cond, curr_map: evaluate_path_length_llm(
            curr_map, cond[1], jnp.divide(NormalizationWeights.PATHLENGTH.value, normal_weights)
        ),  # 2 (diameter)
        lambda cond, curr_map: evaluate_amount_llm(
            curr_map, cond[2], jnp.divide(NormalizationWeights.WALL.value, normal_weights), Dungeon3Tiles.WALL.value
        ),  # 3 (block)
        lambda cond, curr_map: evaluate_amount_llm(
            curr_map, cond[3], jnp.divide(NormalizationWeights.MONSTER.value, normal_weights), Dungeon3Tiles.BAT.value
        ),  # 4 (bat_amount)
        lambda cond, curr_map: evaluate_direction_llm(
            curr_map, cond[4], jnp.divide(NormalizationWeights.DIRECTION.value, normal_weights), Dungeon3Tiles.BAT.value
        ),  # 5 (bat_direction)
        lambda cond, curr_map: 0.0,  # 6+
    ]

    # Map indices to functions using `switch`
    def compute_reward(func_idx, cond_value, _curr_env_map):
        reward_values = jax.vmap(lambda idx: jax.lax.switch(idx, reward_funcs, cond_value, _curr_env_map))(
            func_idx)
        return reward_values

    compute_reward_vmap = vmap(compute_reward, in_axes=(0, 0, 0))
    rewards = compute_reward_vmap(reward_i, condition, curr_env_map)

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
    # embedding = embedding_df.to_numpy()
    # embedding = jnp.pad(
    #     embedding, ((0, 0), (0, 716 - embedding.shape[1])), mode="constant"
    # )

    from instruct_rl.dataclass import Instruct

    df_cond = instruct_df.filter(regex='condition_*')
    condition_df = df_cond.reindex(sorted(df_cond.columns, key=lambda x: int(x.split('_')[-1])), axis=1)
    condition = condition_df.to_numpy()

    print(instruct_df["reward_enum"].to_list())
    reward_enum_list = [[int(digit) for digit in str(num)] for num in instruct_df["reward_enum"].to_list()]
    max_len = 5

    def pad_number(num_list, max_len):
        padded_list = [0] * max_len
        for val in num_list:
            padded_list[val - 1] = val
        return padded_list

    reward_enum = jnp.array([pad_number(x, max_len) for x in reward_enum_list])

    print(reward_enum)

    instruct = Instruct(
        reward_i=reward_enum,
        condition=jnp.array(condition),
        embedding=jnp.array([]),
    )

    # Dummy environment maps
    prev_env_map = jnp.zeros((9, 16, 16))
    curr_env_map = jnp.ones((9, 16, 16))

    # Compute rewards
    rewards = get_reward_batch_llm(
        instruct.reward_i, instruct.condition, prev_env_map, curr_env_map, normal_weights=NormalizationWeights.REGION
    )
    print(rewards)

    print(jnp.sum(rewards, axis=1))

    binary_reward_enum = (reward_enum > 0).astype(int)

    print(binary_reward_enum)