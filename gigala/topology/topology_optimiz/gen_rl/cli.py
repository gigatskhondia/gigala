from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_gen_rl")

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap

from .metrics import calculate_smoothness_metric, count_islands, volume_fraction
from .fem import ProblemConfig
from .pipeline import run_search

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the binary topology optimization pipelines from the command line."
    )
    parser.add_argument(
        "--pipeline-mode",
        default="multistage",
        choices=("multistage", "direct64_exact"),
        help="Pipeline mode. Use direct64_exact for exact-only direct 64x64 search.",
    )
    parser.add_argument("--resolution", type=int, default=64, help="Final grid resolution. Default: 64.")
    parser.add_argument("--volume-target", type=float, default=0.55, help="Target material volume fraction.")
    parser.add_argument("--solver-backend", default="scipy", help="Sparse solver backend. Default: scipy.")
    parser.add_argument(
        "--runtime-budget-hours",
        type=float,
        default=3.0,
        help="Wall-clock budget in hours for the staged search.",
    )
    parser.add_argument(
        "--enable-rl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional MaskablePPO refinement when dependencies are installed.",
    )
    parser.add_argument(
        "--rl-device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Torch device for RL refinement. On Apple silicon, use auto or mps.",
    )
    parser.add_argument(
        "--rl-total-timesteps",
        type=int,
        default=100_000,
        help="Total timesteps for the optional RL refinement stage.",
    )
    parser.add_argument(
        "--rl-archive-top-k",
        type=int,
        default=4,
        help="How many top direct-search archive seeds should get an RL refinement run.",
    )
    parser.add_argument("--coarse-population", type=int, default=128, help="Population size for the 16x16 coarse stage.")
    parser.add_argument("--coarse-generations", type=int, default=250, help="Generations for the 16x16 coarse stage.")
    parser.add_argument("--coarse-elite-count", type=int, default=16, help="Elite count kept between generations.")
    parser.add_argument("--stage32-top-k", type=int, default=8, help="Number of coarse candidates promoted to 32x32.")
    parser.add_argument("--stage64-top-k", type=int, default=2, help="Number of 32x32 candidates promoted to 64x64.")
    parser.add_argument("--local-search-steps32", type=int, default=48, help="Boundary-local search iterations at 32x32.")
    parser.add_argument("--local-search-steps64", type=int, default=96, help="Boundary-local search iterations at 64x64.")
    parser.add_argument("--direct-population", type=int, default=48, help="Population size for direct 64x64 exact search.")
    parser.add_argument("--direct-elite-count", type=int, default=8, help="Elite count kept in direct 64x64 search.")
    parser.add_argument("--direct-offspring-batch", type=int, default=16, help="Offspring evaluated per direct 64x64 batch.")
    parser.add_argument("--direct-archive-size", type=int, default=32, help="Archive size for direct 64x64 search.")
    parser.add_argument(
        "--direct-restart-stagnation-evals",
        type=int,
        default=400,
        help="Restart after this many full64 evals without improvement in direct search.",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Worker count for direct exact batch evaluation. Use auto or an integer.",
    )
    parser.add_argument("--max-full-evals", type=int, default=20_000, help="Full-resolution evaluator budget.")
    parser.add_argument("--max-rl-full-evals", type=int, default=5_000, help="Full-resolution evaluator budget inside RL.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for the staged search.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gen_rl_cli"),
        help="Directory where summary and masks will be saved.",
    )
    return parser


def _summary_payload(config: ProblemConfig, artifacts: Any) -> dict[str, Any]:
    return {
        "config": {
            "pipeline_mode": config.pipeline_mode,
            "resolution": config.resolution,
            "volume_target": config.volume_target,
            "solver_backend": config.solver_backend,
            "runtime_budget_hours": config.runtime_budget_hours,
            "enable_rl": config.enable_rl,
            "rl_device": config.rl_device,
            "rl_total_timesteps": config.rl_total_timesteps,
            "rl_archive_top_k": config.rl_archive_top_k,
            "coarse_population": config.coarse_population,
            "coarse_generations": config.coarse_generations,
            "coarse_elite_count": config.coarse_elite_count,
            "stage32_top_k": config.stage32_top_k,
            "stage64_top_k": config.stage64_top_k,
            "local_search_steps32": config.local_search_steps32,
            "local_search_steps64": config.local_search_steps64,
            "direct_population": config.direct_population,
            "direct_elite_count": config.direct_elite_count,
            "direct_offspring_batch": config.direct_offspring_batch,
            "direct_archive_size": config.direct_archive_size,
            "direct_restart_stagnation_evals": config.direct_restart_stagnation_evals,
            "workers": config.workers,
            "max_full_evals": config.max_full_evals,
            "max_rl_full_evals": config.max_rl_full_evals,
            "random_seed": config.random_seed,
        },
        "runtime_sec": artifacts.runtime,
        "fea_counts": artifacts.fea_counts,
        "warnings": artifacts.warnings,
        "metrics": artifacts.metrics,
    }


def _save_outputs(output_dir: Path, artifacts: Any, config: ProblemConfig) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    if config.pipeline_mode == "direct64_exact":
        seed_path = output_dir / "seed64.npy"
        best_path = output_dir / "best64.npy"
        archive_path = output_dir / "archive.json"
        np.save(seed_path, artifacts.initial_population[0])
        np.save(best_path, artifacts.best64)
        archive_entries = [
            {
                "index": index,
                "volume_fraction": volume_fraction(mask),
                "smoothness": calculate_smoothness_metric(mask),
                "islands": count_islands(mask),
            }
            for index, mask in enumerate(artifacts.archive_best)
        ]
        archive_path.write_text(json.dumps({"entries": archive_entries, "search_trace": artifacts.search_trace}, indent=2))
        saved["seed64"] = str(seed_path)
        saved["best64"] = str(best_path)
        saved["archive"] = str(archive_path)

        final_png = output_dir / "best64.png"
        figure = plt.figure(dpi=100)
        print("\nFinal Cantilever beam design:")
        yellow_material = ListedColormap(["#ffffff", "#ffd84d"])
        plt.imshow(artifacts.best64, cmap=yellow_material, vmin=0, vmax=1)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(final_png, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        saved["best64_png"] = str(final_png)
    else:
        np.save(output_dir / "coarse16.npy", artifacts.coarse16)
        saved["coarse16"] = str(output_dir / "coarse16.npy")
        if artifacts.refined32 is not None:
            np.save(output_dir / "refined32.npy", artifacts.refined32)
            saved["refined32"] = str(output_dir / "refined32.npy")
        if artifacts.refined64 is not None:
            np.save(output_dir / "refined64.npy", artifacts.refined64)
            saved["refined64"] = str(output_dir / "refined64.npy")
            final_png = output_dir / "refined64.png"
            figure = plt.figure(dpi=100)
            print("\nFinal Cantilever beam design:")
            yellow_material = ListedColormap(["#ffffff", "#ffd84d"])
            plt.imshow(artifacts.refined64, cmap=yellow_material, vmin=0, vmax=1)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(final_png, bbox_inches="tight", pad_inches=0.05)
            plt.close(figure)
            saved["refined64_png"] = str(final_png)

    summary = _summary_payload(config, artifacts)
    summary["saved_masks"] = saved
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return {"summary": str(summary_path), **saved}


def _progress_logger(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[gen_rl {timestamp}] {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    workers: str | int
    if isinstance(args.workers, str) and args.workers != "auto":
        workers = int(args.workers)
    else:
        workers = args.workers

    config = ProblemConfig(
        resolution=args.resolution,
        pipeline_mode=args.pipeline_mode,
        volume_target=args.volume_target,
        solver_backend=args.solver_backend,
        runtime_budget_hours=args.runtime_budget_hours,
        enable_rl=args.enable_rl,
        rl_device=args.rl_device,
        rl_total_timesteps=args.rl_total_timesteps,
        rl_archive_top_k=args.rl_archive_top_k,
        coarse_population=args.coarse_population,
        coarse_generations=args.coarse_generations,
        coarse_elite_count=args.coarse_elite_count,
        stage32_top_k=args.stage32_top_k,
        stage64_top_k=args.stage64_top_k,
        local_search_steps32=args.local_search_steps32,
        local_search_steps64=args.local_search_steps64,
        direct_population=args.direct_population,
        direct_elite_count=args.direct_elite_count,
        direct_offspring_batch=args.direct_offspring_batch,
        direct_archive_size=args.direct_archive_size,
        direct_restart_stagnation_evals=args.direct_restart_stagnation_evals,
        workers=workers,
        max_full_evals=args.max_full_evals,
        max_rl_full_evals=args.max_rl_full_evals,
        random_seed=args.random_seed,
    )

    _progress_logger(
        f"CLI launch: resolution={config.resolution}, volume_target={config.volume_target}, "
        f"pipeline_mode={config.pipeline_mode}, runtime_budget_hours={config.runtime_budget_hours}, "
        f"enable_rl={config.enable_rl}, "
        f"rl_device={config.rl_device}, "
        f"output_dir={args.output_dir}"
    )
    artifacts = run_search(config, progress=_progress_logger)
    saved = _save_outputs(args.output_dir, artifacts, config)
    _progress_logger(f"artifacts saved: summary={saved['summary']}")

    terminal_summary = {
        "runtime_sec": round(artifacts.runtime, 2),
        "fea_counts": artifacts.fea_counts,
        "warnings": artifacts.warnings,
        "saved": saved,
    }
    print(json.dumps(terminal_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
