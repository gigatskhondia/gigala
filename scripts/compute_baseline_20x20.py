"""Compute non-RL baselines for the 20x20 cantilever.

This script produces the reference numbers against which the RL-only
pipeline on ``codex/rl-only-20x20`` is compared.  It implements three
cheap baselines that do **not** involve PPO:

1. ``full_solid``      — the trivial all-ones mask the RL pipeline uses
                          as its legacy warm-start (for context; outside
                          the feasibility band for target < 1.0).
2. ``random_at_target`` — sample ``N`` random connected masks whose
                          volume sits in the RL feasibility band and
                          keep the lowest-score feasible one.
3. ``eso_strain_energy`` — a classical Evolutionary Structural
                          Optimisation (ESO) / poor-man's SIMP: start
                          from full-solid, repeatedly remove the cells
                          with the lowest strain-energy density until we
                          reach the target volume, re-solving FEM after
                          each batch.  This is the published baseline a
                          reviewer will intuitively expect RL to beat.

The script is intentionally self-contained: no side-effects on project
state, writes everything under ``runlogs/baseline_20x20_<timestamp>/``.
Usage::

    python3 scripts/compute_baseline_20x20.py \
        --resolution 20 --volume-target 0.55 \
        --seeds 17 42 2026 \
        --random-samples 1000 --eso-step-fraction 0.02

All baselines are evaluated with ``Evaluator.evaluate(mask, "full64")``
so their ``score`` is directly comparable to the RL pipeline's
``summary.json:metrics.best.score``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_repo_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_repo_on_path()

from gigala.topology.topology_optimiz.gen_rl.fem import (  # noqa: E402
    EvalResult,
    Evaluator,
    ProblemConfig,
)
from gigala.topology.topology_optimiz.gen_rl.metrics import (  # noqa: E402
    count_islands,
    retain_components_touching_region,
)


def _build_config(resolution: int, volume_target: float, seed: int) -> ProblemConfig:
    """Build a ProblemConfig that mirrors the RL-only launcher so the
    feasibility band (rl_volume_tolerance) is identical to the one the
    trained policy is evaluated against."""
    return ProblemConfig(
        resolution=resolution,
        pipeline_mode="rl_only_exact",
        volume_target=volume_target,
        enable_rl=False,
        random_seed=seed,
        max_full_evals=10_000,
        max_rl_full_evals=10_000,
        rl_volume_slack_lower=0.05,
        rl_volume_slack_upper=0.05,
        rl_volume_tolerance=0.03,
    )


def _support_load_mask(evaluator: Evaluator, resolution: int) -> np.ndarray:
    setup = evaluator.setups[resolution]
    return np.clip(
        np.asarray(setup.support_mask) + np.asarray(setup.load_mask), 0, 1
    ).astype(np.uint8)


def _result_record(name: str, mask: np.ndarray, result: EvalResult) -> dict[str, Any]:
    return {
        "baseline": name,
        "score": float(result.score),
        "compliance": float(result.compliance),
        "volume_fraction": float(result.volume_fraction),
        "smoothness": int(result.smoothness),
        "islands": int(result.islands),
        "passed_filters": bool(result.passed_filters),
        "invalid_reason": result.invalid_reason,
        "fidelity": result.fidelity,
        "fea_performed": bool(result.fea_performed),
        "mask_sum": int(mask.sum()),
    }


def baseline_full_solid(config: ProblemConfig) -> tuple[np.ndarray, EvalResult]:
    res = int(config.resolution)
    mask = np.ones((res, res), dtype=np.uint8)
    evaluator = Evaluator(config)
    return mask, evaluator.evaluate(mask, "full64")


def baseline_random_at_target(
    config: ProblemConfig,
    *,
    n_samples: int,
) -> tuple[np.ndarray, EvalResult, dict[str, Any]]:
    """Sample ``n_samples`` random connected masks inside the feasibility
    band and keep the lowest-score feasible one.

    Each sample is built by shuffling the removable cells (everything
    outside support+load) and removing them one by one from full-solid
    until we reach the upper-band volume, rejecting any removal that
    would disconnect the mask or drop a support/load cell — this is the
    same procedure used by ``_seed_random_near_target`` in the RL
    pipeline, so the distribution is directly comparable.
    """
    res = int(config.resolution)
    evaluator = Evaluator(config)
    support_load = _support_load_mask(evaluator, res)
    target_vol = float(config.volume_target) + float(config.rl_volume_slack_upper)
    target_cells = int(round(np.clip(target_vol, 0.0, 1.0) * res * res))
    rng = np.random.default_rng(int(config.random_seed))

    best_mask: np.ndarray | None = None
    best_result: EvalResult | None = None
    feasible_count = 0
    for sample_idx in range(n_samples):
        mask = np.ones((res, res), dtype=np.uint8)
        immutable = support_load.astype(bool)
        removable = [
            (r, c) for r in range(res) for c in range(res) if not immutable[r, c]
        ]
        rng.shuffle(removable)
        for r, c in removable:
            if int(mask.sum()) <= target_cells:
                break
            mask[r, c] = 0
            retained = retain_components_touching_region(mask, support_load)
            if int(retained.sum()) < int(mask.sum()) or int(count_islands(retained)) > 1:
                mask[r, c] = 1
            else:
                mask = retained
        result = evaluator.evaluate(mask, "full64")
        if bool(result.passed_filters):
            feasible_count += 1
            if best_result is None or float(result.score) < float(best_result.score):
                best_mask = mask.copy()
                best_result = result
    if best_mask is None or best_result is None:
        # Fall back to the last sample (useful for tiny budgets).
        best_mask = mask
        best_result = result
    stats = {
        "samples_drawn": int(n_samples),
        "feasible_samples": int(feasible_count),
        "feasible_rate": float(feasible_count) / max(1, int(n_samples)),
    }
    return best_mask, best_result, stats


def baseline_eso_strain_energy(
    config: ProblemConfig,
    *,
    step_fraction: float,
    max_iterations: int,
) -> tuple[np.ndarray, EvalResult, dict[str, Any]]:
    """Classical ESO: start from full-solid, iteratively remove the cells
    with lowest strain-energy density until we land in the feasibility
    band, re-solving FEM each iteration.

    ``step_fraction`` controls how aggressive each removal pass is (in
    fraction of the grid size).  After each pass we re-solve FEM, clean
    up disconnected islands, and check whether we hit the feasibility
    band; we stop as soon as the lower volume bound is reached.
    """
    res = int(config.resolution)
    evaluator = Evaluator(config)
    support_load = _support_load_mask(evaluator, res)
    target_vol = float(config.volume_target)
    lower_vol = target_vol - float(config.rl_volume_tolerance)
    upper_vol = target_vol + float(config.rl_volume_tolerance)
    cells_per_step = max(1, int(round(step_fraction * res * res)))

    mask = np.ones((res, res), dtype=np.uint8)
    iterations: list[dict[str, Any]] = []
    last_result: EvalResult | None = None
    for step in range(int(max_iterations)):
        result, fields = evaluator.evaluate_with_fields(mask, "full64")
        last_result = result
        vol = float(np.sum(mask)) / float(res * res)
        iterations.append(
            {
                "step": int(step),
                "volume_fraction": vol,
                "score": float(result.score),
                "passed_filters": bool(result.passed_filters),
            }
        )
        if vol <= upper_vol and vol >= lower_vol and bool(result.passed_filters):
            break
        if vol <= lower_vol:
            break
        if fields.strain_energy_density is None:
            break
        energy = np.asarray(fields.strain_energy_density).reshape(res, res)
        candidates = [
            (float(energy[r, c]), r, c)
            for r in range(res)
            for c in range(res)
            if mask[r, c] == 1 and not support_load[r, c]
        ]
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        removals_applied = 0
        for _, r, c in candidates:
            if removals_applied >= cells_per_step:
                break
            if int(mask.sum()) <= int(round(lower_vol * res * res)):
                break
            mask[r, c] = 0
            retained = retain_components_touching_region(mask, support_load)
            if int(retained.sum()) < int(mask.sum()) or int(count_islands(retained)) > 1:
                mask[r, c] = 1
                continue
            mask = retained
            removals_applied += 1
        if removals_applied == 0:
            break
    final_result = evaluator.evaluate(mask, "full64")
    stats = {
        "iterations": iterations,
        "iterations_run": len(iterations),
        "step_fraction": float(step_fraction),
        "cells_per_step": int(cells_per_step),
    }
    return mask, final_result, stats


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {
        "resolution": int(args.resolution),
        "volume_target": float(args.volume_target),
        "random_samples": int(args.random_samples),
        "eso_step_fraction": float(args.eso_step_fraction),
        "eso_max_iterations": int(args.eso_max_iterations),
        "seeds": {},
    }
    for seed in args.seeds:
        config = _build_config(int(args.resolution), float(args.volume_target), int(seed))
        per_seed: dict[str, Any] = {}

        t0 = time.time()
        fs_mask, fs_result = baseline_full_solid(config)
        per_seed["full_solid"] = {
            **_result_record("full_solid", fs_mask, fs_result),
            "runtime_sec": time.time() - t0,
        }

        t0 = time.time()
        rnd_mask, rnd_result, rnd_stats = baseline_random_at_target(
            config, n_samples=int(args.random_samples)
        )
        per_seed["random_at_target"] = {
            **_result_record("random_at_target", rnd_mask, rnd_result),
            "runtime_sec": time.time() - t0,
            **rnd_stats,
        }

        t0 = time.time()
        eso_mask, eso_result, eso_stats = baseline_eso_strain_energy(
            config,
            step_fraction=float(args.eso_step_fraction),
            max_iterations=int(args.eso_max_iterations),
        )
        per_seed["eso_strain_energy"] = {
            **_result_record("eso_strain_energy", eso_mask, eso_result),
            "runtime_sec": time.time() - t0,
            "iterations_run": eso_stats["iterations_run"],
            "step_fraction": eso_stats["step_fraction"],
            "cells_per_step": eso_stats["cells_per_step"],
        }

        results["seeds"][str(seed)] = per_seed

        if args.output_dir is not None:
            seed_dir = Path(args.output_dir) / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            np.save(seed_dir / "full_solid.npy", fs_mask)
            np.save(seed_dir / "random_at_target.npy", rnd_mask)
            np.save(seed_dir / "eso_strain_energy.npy", eso_mask)
            with open(seed_dir / "eso_iterations.json", "w") as fh:
                json.dump(eso_stats["iterations"], fh, indent=2)
    return results


def _aggregate_across_seeds(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for baseline in ("full_solid", "random_at_target", "eso_strain_energy"):
        scores = [
            float(r[baseline]["score"])
            for r in results["seeds"].values()
            if r.get(baseline) is not None
        ]
        feasible = [
            bool(r[baseline]["passed_filters"])
            for r in results["seeds"].values()
            if r.get(baseline) is not None
        ]
        aggregate[baseline] = {
            "score_best": float(min(scores)) if scores else float("inf"),
            "score_median": float(np.median(scores)) if scores else float("inf"),
            "score_worst": float(max(scores)) if scores else float("inf"),
            "feasible_rate": float(np.mean(feasible)) if feasible else 0.0,
        }
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute non-RL baselines for 20x20 cantilever.")
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--volume-target", type=float, default=0.55)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[17, 42, 2026],
        help="Seeds to evaluate. Each seed produces its own set of baselines.",
    )
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--eso-step-fraction", type=float, default=0.02)
    parser.add_argument("--eso-max-iterations", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save per-seed masks and iteration logs.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the JSON summary. Defaults to output-dir/summary.json when output-dir is set.",
    )
    args = parser.parse_args(argv)

    if args.output_dir is not None:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = run_all(args)
    results["aggregate"] = _aggregate_across_seeds(results)
    results["runtime_sec"] = time.time() - t0

    payload = json.dumps(results, indent=2, default=str)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as fh:
            fh.write(payload)
    elif args.output_dir is not None:
        with open(Path(args.output_dir) / "summary.json", "w") as fh:
            fh.write(payload)

    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
