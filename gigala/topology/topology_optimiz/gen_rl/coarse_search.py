from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage

from .fem import EvalResult, Evaluator, ProblemConfig
from .metrics import ensure_binary, frontier_band, prune_isolated_cells
from .representation import apply_patch, boundary_patch_coordinates


@dataclass
class SearchCandidate:
    mask: np.ndarray
    evaluation: EvalResult


ProgressFn = Callable[[str], None]


def _seed_population(resolution: int, size: int, volume_target: float, rng: np.random.Generator) -> list[np.ndarray]:
    population: list[np.ndarray] = []
    for _ in range(size):
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        mask[0, 0] = 1
        mask[0, -1] = 1
        mask[-1, -1] = 1
        mask[0, :] = 1
        mask[:, -1] = 1

        target_density = rng.uniform(max(volume_target - 0.05, 0.15), min(volume_target + 0.05, 0.80))
        while mask.mean() < target_density:
            frontier = np.logical_and(ndimage.binary_dilation(mask, iterations=1), mask == 0)
            frontier_coords = np.argwhere(frontier)
            if len(frontier_coords) == 0:
                break
            take = max(1, len(frontier_coords) // 6)
            chosen = frontier_coords[rng.choice(len(frontier_coords), size=min(take, len(frontier_coords)), replace=False)]
            for row, col in chosen:
                patch_size = int(rng.choice([1, 2, 3]))
                row_end = min(row + patch_size, resolution)
                col_end = min(col + patch_size, resolution)
                mask[row:row_end, col:col_end] = 1
        mask = ndimage.binary_closing(mask, iterations=1).astype(np.uint8)
        mask = prune_isolated_cells(mask)
        mask[0, 0] = 1
        mask[0, -1] = 1
        mask[-1, -1] = 1
        population.append(mask)
    return population


def _patch_flip(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    patch_size = int(rng.choice([2, 3, 4]))
    resolution = mask.shape[0]
    row = int(rng.integers(0, resolution - patch_size + 1))
    col = int(rng.integers(0, resolution - patch_size + 1))
    mutated = mask.copy()
    mutated[row : row + patch_size, col : col + patch_size] = 1 - mutated[row : row + patch_size, col : col + patch_size]
    return mutated


def _edge_morph(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mutated = mask.copy()
    frontier = frontier_band(mask, width=2)
    material_frontier = np.argwhere(np.logical_and(mask == 1, frontier))
    void_frontier = np.argwhere(np.logical_and(mask == 0, frontier))
    if rng.random() < 0.5 and len(material_frontier) > 0:
        row, col = material_frontier[rng.integers(0, len(material_frontier))]
        mutated[row, col] = 0
    elif len(void_frontier) > 0:
        row, col = void_frontier[rng.integers(0, len(void_frontier))]
        mutated[row, col] = 1
    return mutated


def _volume_preserving_swap(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mutated = mask.copy()
    frontier = frontier_band(mask, width=2)
    material_frontier = np.argwhere(np.logical_and(mask == 1, frontier))
    void_frontier = np.argwhere(np.logical_and(mask == 0, frontier))
    if len(material_frontier) == 0 or len(void_frontier) == 0:
        return mutated
    rem_row, rem_col = material_frontier[rng.integers(0, len(material_frontier))]
    add_row, add_col = void_frontier[rng.integers(0, len(void_frontier))]
    mutated[rem_row, rem_col] = 0
    mutated[add_row, add_col] = 1
    return mutated


def _symmetry_aware_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    symmetry_axis: str | None = None,
) -> np.ndarray:
    resolution = parent_a.shape[0]
    child = parent_a.copy()
    if symmetry_axis == "vertical":
        split = int(rng.integers(1, resolution - 1))
        child[:, split:] = parent_b[:, split:]
        mirrored = np.fliplr(child[:, : split])
        child[:, resolution - mirrored.shape[1] :] = mirrored
    elif symmetry_axis == "horizontal":
        split = int(rng.integers(1, resolution - 1))
        child[split:, :] = parent_b[split:, :]
        mirrored = np.flipud(child[:split, :])
        child[resolution - mirrored.shape[0] :, :] = mirrored
    else:
        split = int(rng.integers(1, resolution - 1))
        child[:, split:] = parent_b[:, split:]
    return child


def _mutate(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    operator = rng.choice(["patch", "edge", "swap"])
    if operator == "patch":
        mutated = _patch_flip(mask, rng)
    elif operator == "edge":
        mutated = _edge_morph(mask, rng)
    else:
        mutated = _volume_preserving_swap(mask, rng)
    mutated[0, :] = 1
    mutated[:, -1] = 1
    mutated[0, 0] = 1
    mutated[0, -1] = 1
    mutated[-1, -1] = 1
    return prune_isolated_cells(mutated)


def run_coarse_search(
    evaluator: Evaluator,
    config: ProblemConfig,
    *,
    resolution: int = 16,
    top_k: int = 8,
    deadline: float | None = None,
    progress: ProgressFn | None = None,
) -> list[SearchCandidate]:
    rng = np.random.default_rng(config.random_seed)
    population = _seed_population(resolution, config.coarse_population, config.volume_target, rng)
    candidates: list[SearchCandidate] = []
    seen: set[bytes] = set()

    log_every = max(1, config.coarse_generations // 10)

    for generation in range(config.coarse_generations):
        scored: list[SearchCandidate] = []
        for mask in population:
            signature = ensure_binary(mask).tobytes()
            if signature in seen:
                continue
            seen.add(signature)
#             evaluation = evaluator.evaluate(mask, "proxy16")
            evaluation = evaluator.evaluate(mask, "proxy32")
            scored.append(SearchCandidate(mask=mask, evaluation=evaluation))

        scored.sort(key=lambda item: item.evaluation.score)
        candidates = scored[:top_k]
        elites = scored[: config.coarse_elite_count]
        if progress and (generation == 0 or (generation + 1) % log_every == 0 or generation + 1 == config.coarse_generations):
            if candidates:
                best = candidates[0].evaluation
                progress(
                    f"coarse16 generation {generation + 1}/{config.coarse_generations}: "
                    f"best_score={best.score:.4f}, volume={best.volume_fraction:.4f}, "
#                     f"proxy16_evals={int(evaluator.fea_counts['proxy16'])}, cache_hits={evaluator.cache_hits}"
                    f"proxy16_evals={int(evaluator.fea_counts['proxy32'])}, cache_hits={evaluator.cache_hits}"
                )
        if deadline and time.time() >= deadline:
            if progress:
                progress("coarse16 deadline reached; stopping coarse search early.")
            break
        if not elites:
            if progress:
                progress("coarse16 produced no elites; stopping coarse search.")
            break

        next_population = [elite.mask.copy() for elite in elites]
        while len(next_population) < config.coarse_population:
            parent_a = elites[int(rng.integers(0, len(elites)))].mask
            parent_b = elites[int(rng.integers(0, len(elites)))].mask
            child = _symmetry_aware_crossover(parent_a, parent_b, rng, symmetry_axis=None)
            child = _mutate(child, rng)
            next_population.append(child)
        population = next_population

    return candidates


def boundary_local_search(
    seeds: list[np.ndarray],
    evaluator: Evaluator,
    config: ProblemConfig,
    *,
    resolution: int,
    fidelity: str,
    patch_sizes: tuple[int, ...],
    top_k: int,
    steps: int,
    deadline: float | None = None,
    progress: ProgressFn | None = None,
) -> list[SearchCandidate]:
    results: list[SearchCandidate] = []
    rng = np.random.default_rng(config.random_seed + resolution + steps)

    log_every = max(1, steps // 5)

    for seed_index, seed in enumerate(seeds, start=1):
        current = ensure_binary(seed)
        current_eval = evaluator.evaluate(current, fidelity)  # type: ignore[arg-type]
        if progress:
            progress(
                f"{fidelity} seed {seed_index}/{len(seeds)}: "
                f"initial_score={current_eval.score:.4f}, fea_performed={current_eval.fea_performed}"
            )
        for step_index in range(steps):
            if deadline and time.time() >= deadline:
                if progress:
                    progress(f"{fidelity} deadline reached during seed {seed_index}; stopping local search early.")
                break
            proposals: list[tuple[np.ndarray, EvalResult]] = []
            for patch_size in patch_sizes:
                for row, col in boundary_patch_coordinates(current, patch_size, width=config.frontier_width)[:24]:
                    patch = current[row : row + patch_size, col : col + patch_size]
                    if patch.all():
                        candidate = apply_patch(current, row, col, patch_size, 0)
                    elif not patch.any():
                        candidate = apply_patch(current, row, col, patch_size, 1)
                    else:
                        continue
                    candidate[0, 0] = 1
                    candidate[0, -1] = 1
                    candidate[-1, -1] = 1
                    candidate = prune_isolated_cells(candidate)
                    proposals.append((candidate, evaluator.evaluate(candidate, fidelity)))  # type: ignore[arg-type]

            if not proposals:
                break
            proposals.sort(key=lambda item: item[1].score)
            best_mask, best_eval = proposals[0]
            if best_eval.score < current_eval.score:
                current = best_mask
                current_eval = best_eval
            elif rng.random() < 0.10:
                current, current_eval = proposals[min(1, len(proposals) - 1)]
            else:
                break
            if progress and ((step_index + 1) % log_every == 0 or step_index + 1 == steps):
                progress(
                    f"{fidelity} seed {seed_index}/{len(seeds)} step {step_index + 1}/{steps}: "
                    f"best_score={current_eval.score:.4f}, "
#                     f"proxy32_evals={int(evaluator.fea_counts['proxy32'])}, "
#                     f"full64_evals={int(evaluator.fea_counts['full64'])}, "
                    f"proxy32_evals={int(evaluator.fea_counts['proxy64'])}, "
                    f"full64_evals={int(evaluator.fea_counts['full128'])}, "
                    f"cache_hits={evaluator.cache_hits}"
                )
        results.append(SearchCandidate(mask=current, evaluation=current_eval))

    results.sort(key=lambda item: item.evaluation.score)
    return results[:top_k]
