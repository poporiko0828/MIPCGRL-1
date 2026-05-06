import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Tiles

from ..direction_type import Direction
from .aggregate_direction import aggregate_direction, generate_direction_map

if __name__ == "__main__":
    sample_map = jnp.zeros((4, 4)).at[0, 0].set(Dungeon3Tiles.BAT.value)

    direction_map = generate_direction_map(jnp.array([Direction.west.value]), 4, 4)
    aggregated = aggregate_direction(
        sample_map,
        Dungeon3Tiles.BAT.value,
        jnp.array([Direction.west.value]),
        4,
        4,
    )

    print(f"direction_map:\n{direction_map}")
    print(f"map:\n{sample_map}")
    print(f"aggregated:\n{aggregated}")
