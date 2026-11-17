from __future__ import annotations

import dataclasses
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import ndimage

from .fem import EvalResult, Evaluator, ProblemConfig, evaluation_snapshot
from .metrics import (
    calculate_smoothness_metric,
    count_islands,
    ensure_binary,
    frontier_band,
    prune_isolated_cells,
    retain_components_touching_region,
    volume_fraction,
)
from .representation import immutable_element_mask


ProgressFn = Callable[[str], None]

_WORKER_EVALUATOR: Evaluator | None = None


@dataclass
class DirectCandidate:
    mask: np.ndarray
    evaluation: EvalResult
    origin: str = "search"


@dataclass
class DirectSearchArtifacts:
    initial_population: list[np.ndarray]
    archive_best: list[np.ndarray]
    best64: np.ndarray
    metrics: dict[str, dict[str, float | int | bool | str | None]]
    fea_counts: dict[str, float]
    runtime: float
    warnings: list[str] = field(default_factory=list)
    search_trace: list[dict[str, float | int | str | bool | None]] = field(default_factory=list)


def _ranking_key(candidate: DirectCandidate) -> tuple[float, float, float, int]:
    evaluation = candidate.evaluation
    feasible_rank = 0.0 if evaluation.passed_filters else 1.0
    return (
        feasible_rank,
        float(evaluation.score),
        float(evaluation.compliance),
        int(evaluation.smoothness),
    )


def _result_to_dict(result: EvalResult) -> dict[str, float | int | bool | str | None]:
    return dataclasses.asdict(result)


def _ensure_immutable(mask: np.ndarray, immutable_mask: np.ndarray) -> np.ndarray:
    binary = ensure_binary(mask)
    binary = np.maximum(binary, immutable_mask)
    return binary.astype(np.uint8)


def _resolve_workers(config: ProblemConfig) -> int:
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file in (None, "<stdin>"):
        return 1
    if isinstance(config.workers, int):
        return max(1, config.workers)
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count // 2))


def _draw_manhattan_path(mask: np.ndarray, start: tuple[int, int], stop: tuple[int, int], rng: np.random.Generator) -> None:
    row, col = start
    target_row, target_col = stop
    while row != target_row or col != target_col:
        mask[row, col] = 1
        moves: list[tuple[int, int]] = []
        if row != target_row:
            moves.append((1 if target_row > row else -1, 0))
        if col != target_col:
            moves.append((0, 1 if target_col > col else -1))
        delta_row, delta_col = moves[int(rng.integers(0, len(moves)))]
        row += delta_row
        col += delta_col
    mask[target_row, target_col] = 1


def _thicken_to_target(mask: np.ndarray, target_volume: float, immutable_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    binary = ensure_binary(mask)
    max_iterations = max(2, binary.shape[0] // 8)
    iterations = 0
    while volume_fraction(binary) < target_volume and iterations < max_iterations:
        dilated = ndimage.binary_dilation(binary, iterations=1).astype(np.uint8)
        binary = _ensure_immutable(dilated, immutable_mask)
        iterations += 1

    carve_probability = 0.04 + 0.04 * float(rng.random())
    carve_mask = rng.random(binary.shape) < carve_probability
    binary = np.where(carve_mask & (immutable_mask == 0), 0, binary).astype(np.uint8)
    binary = prune_isolated_cells(binary)
    return _ensure_immutable(binary, immutable_mask)


def _make_seed(
    resolution: int,
    volume_target: float,
    support_load_mask: np.ndarray,
    immutable_mask: np.ndarray,
    rng: np.random.Generator,
    strategy: str,
) -> np.ndarray:
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    mask = _ensure_immutable(mask, immutable_mask)
    support_points = [tuple(point) for point in np.argwhere(support_load_mask == 1)]
    primary_support = support_points[int(rng.integers(0, len(support_points) - 1))]
    top_right = (0, resolution - 1)
    load = (resolution - 1, resolution - 1)

    if strategy == "path_growth":
        _draw_manhattan_path(mask, load, primary_support, rng)
        _draw_manhattan_path(mask, top_right, (0, 0), rng)
    elif strategy == "bridge":
        _draw_manhattan_path(mask, load, top_right, rng)
        branch_row = int(rng.integers(resolution // 4, resolution - 1))
        branch_col = int(rng.integers(resolution // 4, resolution - 1))
        _draw_manhattan_path(mask, (branch_row, branch_col), primary_support, rng)
    else:
        control_points = [
            load,
            (int(rng.integers(resolution // 3, resolution - 1)), int(rng.integers(resolution // 5, resolution - 1))),
            (int(rng.integers(1, resolution // 2)), int(rng.integers(1, resolution - 1))),
            primary_support,
        ]
        for left, right in zip(control_points, control_points[1:]):
            _draw_manhattan_path(mask, left, right, rng)

    extra_branches = int(rng.integers(1, 4))
    material = np.argwhere(mask == 1)
    for _ in range(extra_branches):
        start_row, start_col = material[int(rng.integers(0, len(material)))]
        stop = (int(rng.integers(0, resolution)), int(rng.integers(0, resolution)))
        _draw_manhattan_path(mask, (int(start_row), int(start_col)), stop, rng)

    thickened = _thicken_to_target(
        mask,
        target_volume=float(np.clip(volume_target + rng.uniform(-0.04, 0.06), 0.20, 0.85)),
        immutable_mask=immutable_mask,
        rng=rng,
    )
    thickened = retain_components_touching_region(thickened, support_load_mask)
    return _ensure_immutable(thickened, immutable_mask)


def generate_initial_population(
    config: ProblemConfig,
    evaluator: Evaluator,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    resolution = config.resolution
    setup = evaluator.setups[resolution]
    support_load_mask = np.clip(setup.support_mask + setup.load_mask, 0, 1).astype(np.uint8)
    immutable_mask = np.maximum(immutable_element_mask(resolution), support_load_mask)
    strategies = ("path_growth", "bridge", "skeleton")
    return [
        _make_seed(
            resolution,
            config.volume_target,
            support_load_mask,
            immutable_mask,
            rng,
            strategies[index % len(strategies)],
        )
        for index in range(config.direct_population)
    ]


def _choose_mutable_coordinate(mask: np.ndarray, immutable_mask: np.ndarray, rng: np.random.Generator, *, value: int | None = None) -> tuple[int, int] | None:
    candidates = np.argwhere(immutable_mask == 0)
    if value is not None:
        candidates = np.array([point for point in candidates if mask[tuple(point)] == value], dtype=int)
    if len(candidates) == 0:
        return None
    row, col = candidates[int(rng.integers(0, len(candidates)))]
    return int(row), int(col)


def _single_cell_flip(mask: np.ndarray, immutable_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mutated = mask.copy()
    coordinate = _choose_mutable_coordinate(mutated, immutable_mask, rng)
    if coordinate is None:
        return mutated
    row, col = coordinate
    mutated[row, col] = 1 - mutated[row, col]
    return mutated


def _volume_preserving_swap(mask: np.ndarray, immutable_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mutated = mask.copy()
    remove_coord = _choose_mutable_coordinate(mutated, immutable_mask, rng, value=1)
    add_coord = _choose_mutable_coordinate(mutated, immutable_mask, rng, value=0)
    if remove_coord is None or add_coord is None:
        return mutated
    mutated[remove_coord] = 0
    mutated[add_coord] = 1
    return mutated


def _patch_toggle(mask: np.ndarray, immutable_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mutated = mask.copy()
    size = int(rng.choice([2, 3, 4]))
    limit = mutated.shape[0] - size + 1
    row = int(rng.integers(0, limit))
    col = int(rng.integers(0, limit))
    immutable_patch = immutable_mask[row : row + size, col : col + size]
    toggle = immutable_patch == 0
    mutated[row : row + size, col : col + size] = np.where(
        toggle,
        1 - mutated[row : row + size, col : col + size],
        mutated[row : row + size, col : col + size],
    )
    return mutated


def _path_add(mask: np.ndarray, immutable_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mutated = mask.copy()
    frontier = frontier_band(mutated, width=1)
    start_candidates = np.argwhere(frontier | (mutated == 1))
    if len(start_candidates) == 0:
        return mutated
    row, col = start_candidates[int(rng.integers(0, len(start_candidates)))]
    length = int(rng.integers(4, 14))
    for _ in range(length):
        mutated[row, col] = 1
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        delta_row, delta_col = moves[int(rng.integers(0, len(moves)))]
        row = int(np.clip(row + delta_row, 0, mutated.shape[0] - 1))
        col = int(np.clip(col + delta_col, 0, mutated.shape[1] - 1))
    return _ensure_immutable(mutated, immutable_mask)


def _path_remove(mask: np.ndarray, immutable_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mutated = mask.copy()
    material_frontier = np.argwhere((mutated == 1) & frontier_band(mutated, width=1) & (immutable_mask == 0))
    if len(material_frontier) == 0:
        return mutated
    row, col = material_frontier[int(rng.integers(0, len(material_frontier)))]
    length = int(rng.integers(3, 10))
    for _ in range(length):
        if immutable_mask[row, col] == 0:
            mutated[row, col] = 0
        neighbors = []
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt_row = row + delta_row
            nxt_col = col + delta_col
            if (
                0 <= nxt_row < mutated.shape[0]
                and 0 <= nxt_col < mutated.shape[1]
                and mutated[nxt_row, nxt_col] == 1
                and immutable_mask[nxt_row, nxt_col] == 0
            ):
                neighbors.append((nxt_row, nxt_col))
        if not neighbors:
            break
        row, col = neighbors[int(rng.integers(0, len(neighbors)))]
    return mutated


def _bridge_add(mask: np.ndarray, immutable_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mutated = mask.copy()
    material = np.argwhere(mutated == 1)
    if len(material) < 2:
        return mutated
    left = material[int(rng.integers(0, len(material)))]
    right = material[int(rng.integers(0, len(material)))]
    _draw_manhattan_path(mutated, (int(left[0]), int(left[1])), (int(right[0]), int(right[1])), rng)
    return _ensure_immutable(mutated, immutable_mask)


def _thicken(mask: np.ndarray, immutable_mask: np.ndarray) -> np.ndarray:
    mutated = ndimage.binary_dilation(mask, iterations=1).astype(np.uint8)
    return _ensure_immutable(mutated, immutable_mask)


def _thin(mask: np.ndarray, immutable_mask: np.ndarray) -> np.ndarray:
    mutated = ndimage.binary_erosion(mask, iterations=1).astype(np.uint8)
    return _ensure_immutable(mutated, immutable_mask)


def _cleanup(mask: np.ndarray, immutable_mask: np.ndarray, support_load_mask: np.ndarray) -> np.ndarray:
    cleaned = prune_isolated_cells(mask)
    cleaned = retain_components_touching_region(cleaned, support_load_mask)
    return _ensure_immutable(cleaned, immutable_mask)


def _crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator, immutable_mask: np.ndarray) -> np.ndarray:
    child = parent_a.copy()
    kind = str(rng.choice(["vertical", "horizontal", "uniform"]))
    if kind == "vertical":
        split = int(rng.integers(1, parent_a.shape[1] - 1))
        child[:, split:] = parent_b[:, split:]
    elif kind == "horizontal":
        split = int(rng.integers(1, parent_a.shape[0] - 1))
        child[split:, :] = parent_b[split:, :]
    else:
        chooser = rng.random(parent_a.shape) > 0.5
        child = np.where(chooser, parent_a, parent_b).astype(np.uint8)
    return _ensure_immutable(child, immutable_mask)


def mutate_mask(
    mask: np.ndarray,
    *,
    immutable_mask: np.ndarray,
    support_load_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    operator = str(
        rng.choice(
            [
                "single_cell_flip",
                "volume_preserving_swap",
                "patch_toggle",
                "path_add",
                "path_remove",
                "bridge_add",
                "thicken",
                "thin",
                "cleanup",
            ]
        )
    )
    if operator == "single_cell_flip":
        mutated = _single_cell_flip(mask, immutable_mask, rng)
    elif operator == "volume_preserving_swap":
        mutated = _volume_preserving_swap(mask, immutable_mask, rng)
    elif operator == "patch_toggle":
        mutated = _patch_toggle(mask, immutable_mask, rng)
    elif operator == "path_add":
        mutated = _path_add(mask, immutable_mask, rng)
    elif operator == "path_remove":
        mutated = _path_remove(mask, immutable_mask, rng)
    elif operator == "bridge_add":
        mutated = _bridge_add(mask, immutable_mask, rng)
    elif operator == "thicken":
        mutated = _thicken(mask, immutable_mask)
    elif operator == "thin":
        mutated = _thin(mask, immutable_mask)
    else:
        mutated = mask.copy()
    return _cleanup(mutated, immutable_mask, support_load_mask)


def build_mutation_coverage(resolution: int) -> np.ndarray:
    return np.ones((resolution, resolution), dtype=bool)


def _init_worker(config: ProblemConfig) -> None:
    global _WORKER_EVALUATOR
    worker_config = dataclasses.replace(config, max_full_evals=max(config.max_full_evals, 1_000_000))
    _WORKER_EVALUATOR = Evaluator(worker_config)


def _evaluate_full64_worker(mask: np.ndarray) -> EvalResult:
    if _WORKER_EVALUATOR is None:  # pragma: no cover - protected by initializer
        raise RuntimeError("worker evaluator was not initialized")
    return _WORKER_EVALUATOR.evaluate(mask, "full64")


def evaluate_exact_batch(
    masks: Sequence[np.ndarray],
    *,
    evaluator: Evaluator,
    config: ProblemConfig,
    workers: int,
    pool: ProcessPoolExecutor | None = None,
) -> list[DirectCandidate]:
    results: list[DirectCandidate] = []
    pending_masks: list[np.ndarray] = []
    seen_keys: set[tuple[str, str]] = set()

    for mask in masks:
        cache_key, processed, setup = evaluator.cache_key_for(mask, "full64")
        if cache_key in seen_keys:
            continue
        seen_keys.add(cache_key)
        cached = evaluator.cached_result(cache_key)
        if cached is not None:
            results.append(DirectCandidate(mask=processed, evaluation=cached, origin="cache"))
            continue
        invalid = evaluator.screen_processed(processed, "full64", setup=setup)
        if invalid is not None:
            evaluator.store_result(processed, "full64", invalid)
            results.append(DirectCandidate(mask=processed, evaluation=invalid, origin="filtered"))
            continue
        if int(evaluator.fea_counts["full64"]) + len(pending_masks) >= config.max_full_evals:
            budget_result = evaluator.make_invalid_result(processed, "full64", "full_eval_budget_exhausted")
            evaluator.store_result(processed, "full64", budget_result)
            results.append(DirectCandidate(mask=processed, evaluation=budget_result, origin="budget"))
            continue
        pending_masks.append(processed)

    if not pending_masks:
        return sorted(results, key=_ranking_key)

    if pool is None or workers <= 1:
        for processed in pending_masks:
            evaluation = evaluator.evaluate(processed, "full64")
            results.append(DirectCandidate(mask=processed, evaluation=evaluation, origin="exact"))
        return sorted(results, key=_ranking_key)

    batch_results = list(pool.map(_evaluate_full64_worker, pending_masks))
    for processed, evaluation in zip(pending_masks, batch_results):
        if evaluation.fea_performed:
            evaluator.fea_counts["full64"] += 1
        evaluator.store_result(processed, "full64", evaluation)
        results.append(DirectCandidate(mask=processed, evaluation=evaluation, origin="exact"))
    return sorted(results, key=_ranking_key)


def _select_best(candidates: Sequence[DirectCandidate], count: int) -> list[DirectCandidate]:
    return sorted(candidates, key=_ranking_key)[:count]


def _merge_archive(archive: Sequence[DirectCandidate], candidates: Sequence[DirectCandidate], count: int) -> list[DirectCandidate]:
    unique: dict[bytes, DirectCandidate] = {}
    for candidate in [*archive, *candidates]:
        signature = ensure_binary(candidate.mask).tobytes()
        current = unique.get(signature)
        if current is None or _ranking_key(candidate) < _ranking_key(current):
            unique[signature] = candidate
    return _select_best(list(unique.values()), count)


def run_direct_search_core(config: ProblemConfig, *, progress: ProgressFn | None = None) -> DirectSearchArtifacts:
    start = time.time()
    deadline = start + config.runtime_budget_hours * 3600.0 if config.has_runtime_budget() else None
    evaluator = Evaluator(config)
    rng = np.random.default_rng(config.random_seed)
    warnings: list[str] = []
    workers = _resolve_workers(config)
    if workers == 1 and config.workers != 1 and getattr(sys.modules.get("__main__"), "__file__", None) in (None, "<stdin>"):
        warnings.append("Falling back to serial exact evaluation because multiprocessing is unsafe from an interactive entrypoint.")
    setup = evaluator.setups[config.resolution]
    support_load_mask = np.clip(setup.support_mask + setup.load_mask, 0, 1).astype(np.uint8)
    immutable_mask = np.maximum(immutable_element_mask(config.resolution), support_load_mask)

    if progress:
        progress(
            f"direct64 exact search started: resolution={config.resolution}, population={config.direct_population}, "
            f"offspring_batch={config.direct_offspring_batch}, workers={workers}, enable_rl={config.enable_rl}"
        )
        if warnings:
            progress(warnings[-1])

    initial_population = generate_initial_population(config, evaluator, rng)
    initial_population_snapshot = [mask.copy() for mask in initial_population]

    trace: list[dict[str, float | int | str | bool | None]] = []

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(config,)) if workers > 1 else nullcontext() as maybe_pool:  # type: ignore[arg-type]
        pool = maybe_pool if workers > 1 else None
        population = evaluate_exact_batch(initial_population, evaluator=evaluator, config=config, workers=workers, pool=pool)
        if not population:
            raise RuntimeError("Direct exact search could not evaluate initial population.")
        population = _select_best(population, config.direct_population)
        archive = _merge_archive([], population, config.direct_archive_size)
        best = population[0]
        initial_best = population[0]
        last_improvement_eval = int(evaluator.fea_counts["full64"])

        trace.append(
            {
                "event": "initial_population",
                "full64_evals": int(evaluator.fea_counts["full64"]),
                "best_score": float(best.evaluation.score),
                "best_volume": float(best.evaluation.volume_fraction),
                "best_islands": int(best.evaluation.islands),
            }
        )

        while (deadline is None or time.time() < deadline) and int(evaluator.fea_counts["full64"]) < config.max_full_evals:
            elites = _select_best(population, max(1, config.direct_elite_count))
            offspring_masks: list[np.ndarray] = []
            while len(offspring_masks) < config.direct_offspring_batch and (deadline is None or time.time() < deadline):
                if rng.random() < 0.75 and len(elites) >= 2:
                    parent_a = elites[int(rng.integers(0, len(elites)))].mask
                    parent_b = elites[int(rng.integers(0, len(elites)))].mask
                    child = _crossover(parent_a, parent_b, rng, immutable_mask)
                else:
                    parent = elites[int(rng.integers(0, len(elites)))].mask
                    child = parent.copy()
                child = mutate_mask(child, immutable_mask=immutable_mask, support_load_mask=support_load_mask, rng=rng)
                offspring_masks.append(child)

            offspring = evaluate_exact_batch(offspring_masks, evaluator=evaluator, config=config, workers=workers, pool=pool)
            if not offspring:
                break

            population = _select_best([*population, *offspring], config.direct_population)
            archive = _merge_archive(archive, offspring, config.direct_archive_size)

            if _ranking_key(population[0]) < _ranking_key(best):
                best = population[0]
                last_improvement_eval = int(evaluator.fea_counts["full64"])

            trace.append(
                {
                    "event": "offspring_batch",
                    "full64_evals": int(evaluator.fea_counts["full64"]),
                    "best_score": float(best.evaluation.score),
                    "best_compliance": float(best.evaluation.compliance),
                    "best_volume": float(best.evaluation.volume_fraction),
                    "population_best_smoothness": int(population[0].evaluation.smoothness),
                }
            )

            if progress and (len(trace) == 1 or len(trace) % 5 == 0):
                progress(
                    f"direct64 batch {len(trace) - 1}: best_score={best.evaluation.score:.4f}, "
                    f"compliance={best.evaluation.compliance:.4f}, volume={best.evaluation.volume_fraction:.4f}, "
                    f"full64_evals={int(evaluator.fea_counts['full64'])}, cache_hits={evaluator.cache_hits}"
                )

            if int(evaluator.fea_counts["full64"]) - last_improvement_eval >= config.direct_restart_stagnation_evals:
                restarted_masks = generate_initial_population(config, evaluator, rng)
                restarted = evaluate_exact_batch(restarted_masks, evaluator=evaluator, config=config, workers=workers, pool=pool)
                protected = _select_best(population, max(1, config.direct_elite_count))
                population = _select_best([*protected, *restarted], config.direct_population)
                archive = _merge_archive(archive, restarted, config.direct_archive_size)
                last_improvement_eval = int(evaluator.fea_counts["full64"])
                warnings.append("direct64 search restarted after stagnation")
                trace.append(
                    {
                        "event": "restart",
                        "full64_evals": int(evaluator.fea_counts["full64"]),
                        "best_score": float(population[0].evaluation.score),
                        "best_volume": float(population[0].evaluation.volume_fraction),
                    }
                )
                if progress:
                    progress(
                        f"direct64 stagnation restart: best_score={population[0].evaluation.score:.4f}, "
                        f"full64_evals={int(evaluator.fea_counts['full64'])}"
                    )

            if int(evaluator.fea_counts["full64"]) >= config.max_full_evals:
                warnings.append("direct64 full64 evaluation budget exhausted")
                break

    runtime = time.time() - start
    metrics = {
        "seed64": _result_to_dict(initial_best.evaluation),
        "best64": _result_to_dict(best.evaluation),
        "archive_summary": {
            "best_score": float(archive[0].evaluation.score),
            "best_volume": float(archive[0].evaluation.volume_fraction),
            "entry_count": len(archive),
            "best_smoothness": int(archive[0].evaluation.smoothness),
            "best_islands": int(archive[0].evaluation.islands),
        },
    }
    if progress:
        progress(
            f"direct64 exact search finished: runtime_sec={runtime:.2f}, "
            f"best_score={best.evaluation.score:.4f}, full64_evals={int(evaluator.fea_counts['full64'])}, "
            f"cache_hits={evaluator.cache_hits}"
        )
    return DirectSearchArtifacts(
        initial_population=initial_population_snapshot,
        archive_best=[candidate.mask.copy() for candidate in archive],
        best64=best.mask.copy(),
        metrics=metrics,
        fea_counts=evaluation_snapshot(evaluator),
        runtime=runtime,
        warnings=warnings,
        search_trace=trace,
    )


class nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> bool:
        return False
