import logging
import os
import shutil
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from os.path import dirname, join, basename
from typing import Tuple

from flax import struct

from envs.pathfinding import calc_path_from_a_to_b, FloodPath, calc_path_length
from envs.probs.dungeon3 import Dungeon3Tiles, Dungeon3Problem
from envs.solution import get_solution_jit
from pcgrllm.evaluation.base import LevelEvaluator, EvaluationResult
from pcgrllm.utils.cuda import get_cuda_version
from pcgrllm.utils.storage import Iteration
from pcgrllm.utils.solution_evaluator import create_fixed_size_onehot

NOT_EXISTS = jnp.array([-1, -1])

ENUM_TO_ENCOUNTER_INDEX = {
    Dungeon3Tiles.BAT: 0,
    Dungeon3Tiles.SCORPION: 1,
}

tile_keys = jnp.array(list(ENUM_TO_ENCOUNTER_INDEX.keys()))
tile_indices = jnp.array(list(ENUM_TO_ENCOUNTER_INDEX.values()))

CUDA_VERSION = get_cuda_version()


@struct.dataclass
class EvaluationResultStruct:
    similarity: float = 0
    diversity: float = 0

    # Scenario generation task
    playability: float = 0  # If the player can reach to the door
    path_length: float = 0  # The path length from the player to the door
    solvability: float = 0  # if the player can reach the door with the key
    n_solutions: float = 0  # Number of solutions the player can reach the door
    loss_solutions: float = (
        0  # Loss of between the target and the current solution count
    )
    reach_imp_perc: float = 0  # (reachable of important tiles <-> prompt)
    exist_imp_perc: float = 0  # (existence of important tiles <-> prompt)

    acc_imp_perc: float = 0
    fp_imp_perc: float = 0
    fn_imp_perc: float = 0
    tp_imp_perc: float = 0
    tn_imp_perc: float = 0


def eval_level(level: np.ndarray, scenario_num) -> Tuple[float, float]:
    """
    # Scenario generation task
    playability: float = 0  # If the player can reach to the door
    path_length: float = 0  # The path length from the player to the door
    solvability: float = 0  # if the player can reach the door with the key
    n_solutions: float = 0  # Number of solutions the player can reach the door
    loss_solutions: float = 0  # Loss of between the target and the current solution count
    acc_imp_tiles: float = 0  # (reachable of important tiles <-> prompt)
    exist_imp_tiles: float = 0  # (existence of important tiles <-> prompt)
    """
    # if level shape is not 2d, raise error
    if len(level.shape) != 2:
        raise ValueError(f"The level must be 2D array. Got {len(level.shape)}D array.")
    #
    passable_tiles = Dungeon3Problem.passable_tiles
    imp_tiles = ScenarioPreset().scenarios[str(scenario_num)].important_tiles
    imp_tiles = jnp.array(imp_tiles)

    p_xy = jnp.argwhere(level == Dungeon3Tiles.PLAYER, size=1, fill_value=-1)[0]
    d_xy = jnp.argwhere(level == Dungeon3Tiles.DOOR, size=1, fill_value=-1)[0]
    k_xy = jnp.argwhere(level == Dungeon3Tiles.KEY, size=1, fill_value=-1)[0]

    p_t_length, _, _ = calc_path_from_a_to_b(level, passable_tiles, p_xy, d_xy)
    p_t_connected = jnp.where(p_t_length >= 0, 1, 0)

    p_k_length = jax.lax.cond(
        jnp.all(k_xy != NOT_EXISTS),
        lambda _: calc_path_from_a_to_b(level, passable_tiles, p_xy, k_xy)[0],
        lambda _: jnp.array(-1).astype(float),
        operand=None,
    )

    p_k_connected = jnp.where(p_k_length > 0, 1, 0)
    is_solavable = jnp.where(jnp.logical_and(p_t_connected, p_k_connected), 1, 0)

    def process_important_tile(time_num):
        _xy = jnp.argwhere(level == time_num, size=1, fill_value=-1)[0]

        def tile_exists(_xy):
            _passable_tiles = jnp.append(Dungeon3Problem.passable_tiles, time_num)
            flood_path_net = FloodPath()
            flood_path_net.init_params(level.shape)
            path_length, _, _ = calc_path_length(
                flood_path_net, level, _passable_tiles, Dungeon3Tiles.PLAYER, time_num
            )
            n_acc = jnp.where(path_length > 0, 1, 0)
            return 1, n_acc  # Tile exists and may be reachable

        def tile_not_exists(_xy):
            return 0, 0  # Tile does not exist

        return jax.lax.cond(
            (_xy != NOT_EXISTS).all(), tile_exists, tile_not_exists, _xy
        )

    if CUDA_VERSION is None or CUDA_VERSION > 12:
        imp_tile_results = jax.lax.map(process_important_tile, imp_tiles)
    else:
        imp_tile_results = jax.vmap(process_important_tile)(jnp.array(imp_tiles))

    # Summing results
    n_exist_imp_tiles = jnp.sum(
        imp_tile_results[0]
    )  # Count of existing important tiles
    n_reach_imp_tiles = jnp.sum(
        imp_tile_results[1]
    )  # Count of reachable important tiles

    solutions = get_solution_jit(level)
    enemy_counter = solutions.enemy_encounter
    enemy_counter_type = jnp.any(jnp.where(enemy_counter > 0, 1, 0), axis=0)

    onehot_imp_tiles = create_fixed_size_onehot(imp_tiles).astype(jnp.bool_)

    correct_count = jax.lax.cond(
        solutions.n > 0,
        lambda _: jnp.sum(onehot_imp_tiles == enemy_counter_type),
        lambda _: 0,
        operand=None,
    )

    # True Positive: Predicted as 1 (positive) and actually 1 (positive)
    true_positive = jnp.sum(
        jnp.logical_and(onehot_imp_tiles == 1, enemy_counter_type == 1)
    )
    # True Negative: Predicted as 0 (negative) and actually 0 (negative)
    true_negative = jax.lax.cond(
        solutions.n > 0,
        lambda _: jnp.sum(
            jnp.logical_and(onehot_imp_tiles == 0, enemy_counter_type == 0)
        ),
        lambda _: 0,
        operand=None,
    )

    # False Negative: Predicted as 0 (negative) but actually 1 (positive)
    false_negative = jax.lax.cond(
        solutions.n > 0,
        lambda _: jnp.sum(
            jnp.logical_and(onehot_imp_tiles == 1, enemy_counter_type == 0)
        ),
        lambda _: jnp.sum(enemy_counter_type == 0),
        operand=None,
    )
    # False Positive: Predicted as 1 (positive) but actually 0 (negative)
    false_positive = jnp.sum(
        jnp.logical_and(onehot_imp_tiles == 0, enemy_counter_type == 1)
    )

    # check if the player and door is connected
    playability = p_t_connected
    path_length = p_t_length
    solvability = is_solavable
    n_solutions = solutions.n
    loss_solutions = jnp.abs(len(imp_tiles) - n_solutions)
    reach_imp_tiles = n_reach_imp_tiles / len(imp_tiles)
    exist_imp_tiles = n_exist_imp_tiles / len(imp_tiles)

    correct_count = correct_count / 3
    false_positive = false_positive
    false_negative = false_negative
    true_positive = true_positive
    true_negative = true_negative

    # jax.debug.print("{}, {}, {}, {}", false_positive, false_negative, true_positive, true_negative)

    return EvaluationResultStruct(
        playability=playability,
        path_length=path_length,
        solvability=solvability,
        n_solutions=n_solutions,
        loss_solutions=loss_solutions,
        reach_imp_perc=reach_imp_tiles,
        exist_imp_perc=exist_imp_tiles,
        acc_imp_perc=correct_count,
        fp_imp_perc=false_positive,
        fn_imp_perc=false_negative,
        tp_imp_perc=true_positive,
        tn_imp_perc=true_negative,
    )


def eval_level_jax(levels, scenario_num):
    def eval_level_wrapper(level):
        return eval_level(level, scenario_num=scenario_num)

    if CUDA_VERSION is None or CUDA_VERSION > 12:
        results = jax.vmap(eval_level_wrapper)(levels)
    else:
        results = jax.lax.map(eval_level_wrapper, levels)

    return (
        results.playability,
        results.path_length,
        results.solvability,
        results.n_solutions,
        results.loss_solutions,
        results.reach_imp_perc,
        results.exist_imp_perc,
        results.acc_imp_perc,
        results.fp_imp_perc,
        results.fn_imp_perc,
        results.tp_imp_perc,
        results.tn_imp_perc,
    )


class SolutionEvaluator(LevelEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(
            self,
            iteration: Iteration,
            scenario_num: str = None,
            target_character: str = None,
            visualize: bool = False,
            use_train: bool = False,
            step_filter=None,
    ) -> EvaluationResult:
        numpy_files = iteration.get_numpy_files(
            train=use_train, step_filter=step_filter
        )

        scenario_num = scenario_num if scenario_num is not None else target_character

        levels = jnp.array([numpy_file.load() for numpy_file in numpy_files])

        eval_results = eval_level_jax(levels=levels, scenario_num=scenario_num)

        (
            playability,
            path_length,
            solvability,
            n_solutions,
            loss_solutions,
            reach_imp_tiles,
            exist_imp_tiles,
            acc_imp_perc,
            fp_imp_perc,
            fn_imp_perc,
            tp_imp_perc,
            tn_imp_perc,
        ) = eval_results

        playability = jnp.mean(playability)
        path_length = jnp.nan_to_num(
            jnp.nanmean(jnp.where(path_length != -1, path_length, jnp.nan))
        )
        solvability = jnp.mean(solvability)
        n_solutions = jnp.mean(n_solutions)
        loss_solutions = jnp.mean(loss_solutions)
        reach_imp_tiles = jnp.mean(reach_imp_tiles)
        exist_imp_tiles = jnp.mean(exist_imp_tiles)

        acc_imp_perc = jnp.mean(acc_imp_perc)
        fp_imp_perc = jnp.mean(fp_imp_perc)
        fn_imp_perc = jnp.mean(fn_imp_perc)
        tp_imp_perc = jnp.mean(tp_imp_perc)
        tn_imp_perc = jnp.mean(tn_imp_perc)

        sample_size = len(levels)

        return EvaluationResult(
            task=self.task,
            playability=playability,
            path_length=path_length,
            solvability=solvability,
            n_solutions=n_solutions,
            loss_solutions=loss_solutions,
            reach_imp_perc=reach_imp_tiles,
            exist_imp_perc=exist_imp_tiles,
            acc_imp_perc=acc_imp_perc,
            fp_imp_perc=fp_imp_perc,
            fn_imp_perc=fn_imp_perc,
            tp_imp_perc=tp_imp_perc,
            tn_imp_perc=tn_imp_perc,
            sample_size=sample_size,
        )


# Example


if __name__ == "__main__":
    # Initialize logger
    from debug.scenario_levels import AllLevels

    logger = logging.getLogger(basename(__file__))
    logger.setLevel(logging.DEBUG)

    evaluator = SolutionEvaluator(logger=logger, task=TaskType.Scenario)

    base_path = join(dirname(__file__), "example")
    # Define the path for the iteration folder
    example_path = join(base_path, "scenario_1", "iteration_1")

    # Load the iteration
    iteration = Iteration.from_path(path=example_path)

    numpy_dir = iteration.get_numpy_dir()

    if os.path.exists(numpy_dir):
        shutil.rmtree(numpy_dir)
        os.makedirs(numpy_dir)

    # save the alllevels into the numpy dir
    for idx, level in enumerate(AllLevels[:]):
        np.save(join(iteration.get_numpy_dir(), f"level_{idx}.npy"), level)
    # Run the evaluator with visualization enabled/disabled
    result = evaluator.run(iteration=iteration, scenario_num="2", visualize=True)
    print(result)
