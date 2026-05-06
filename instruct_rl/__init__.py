NUM_CONDITIONS = 5


#         n_region = aggregate_region(env_map)
#         n_path_length = aggregate_path_length(env_map)
#         n_block = aggregate_amount(env_map, Dungeon3Tiles.WALL.value)
#         n_bat = aggregate_amount(env_map, Dungeon3Tiles.BAT.value)
#         n_scorpion = aggregate_amount(env_map, Dungeon3Tiles.SCORPION.value)
#         n_bat_dir0 = aggregate_direction(env_map, Dungeon3Tiles.BAT.value, jnp.array(0), rows, cols)
#         n_bat_dir1 = aggregate_direction(env_map, Dungeon3Tiles.BAT.value, jnp.array(1), rows, cols)
#         n_bat_dir2 = aggregate_direction(env_map, Dungeon3Tiles.BAT.value, jnp.array(2), rows, cols)
#         n_bat_dir3 = aggregate_direction(env_map, Dungeon3Tiles.BAT.value, jnp.array(3), rows, cols)
#         n_scorpion_dir0 = aggregate_direction(env_map, Dungeon3Tiles.SCORPION.value, jnp.array(0), rows, cols)
#         n_scorpion_dir1 = aggregate_direction(env_map, Dungeon3Tiles.SCORPION.value, jnp.array(1), rows, cols)
#         n_scorpion_dir2 = aggregate_direction(env_map, Dungeon3Tiles.SCORPION.value, jnp.array(2), rows, cols)
#         n_scorpion_dir3 = aggregate_direction(env_map, Dungeon3Tiles.SCORPION.value, jnp.array(3), rows, cols)
FEATURE_NAMES = ['region', 'plength',
                 'nblock', 'nbat',
                 'batdir0', 'batdir1', 'batdir2', 'batdir3']

NUM_FEATURES = len(FEATURE_NAMES)