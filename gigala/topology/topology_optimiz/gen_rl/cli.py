from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
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
        choices=("multistage", "direct64_exact", "rl_only_exact"),
        help="Pipeline mode. Use direct64_exact for exact-only direct search, or rl_only_exact for pure RL with no GA.",
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
    parser.add_argument(
        "--rl-boundary-depth",
        type=int,
        default=1,
        help="Boundary contour depth for direct RL removals. Default: 1.",
    )
    parser.add_argument(
        "--rl-stress-hotspot-quantile",
        type=float,
        default=0.95,
        help="Exact stress quantile threshold for direct RL hotspot cells. Default: 0.95.",
    )
    parser.add_argument(
        "--rl-stress-hotspot-dilate",
        type=int,
        default=1,
        help="4-neighbor dilation radius for direct RL hotspot cells. Default: 1.",
    )
    parser.add_argument(
        "--rl-stop-penalty",
        type=float,
        default=0.05,
        help="Penalty for an early stop without any accepted RL improvements. Default: 0.05.",
    )
    parser.add_argument(
        "--rl-degenerate-episode-window",
        type=int,
        default=32,
        help="Stop direct RL training early after this many immediate stop-only episodes. Default: 32.",
    )
    parser.add_argument(
        "--rl-sparse-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use sparse terminal reward (harmonic mean 2*a1*a2/(a1+a2)) with a stop action "
            "and volume-aware action masking. Recommended for rl_only_exact."
        ),
    )
    parser.add_argument(
        "--rl-n-envs",
        type=int,
        default=1,
        help="Number of parallel RL envs (SubprocVecEnv). Default: 1.",
    )
    parser.add_argument(
        "--rl-inference-rollouts",
        type=int,
        default=1,
        help="Rollouts at RL inference time (1 deterministic + N-1 stochastic). Default: 1.",
    )
    parser.add_argument(
        "--rl-policy-size",
        default="small",
        choices=("small", "large"),
        help="Policy network size. 'large' = CNN 32/64/128 + feat_dim=256 + MLP [256,256].",
    )
    parser.add_argument(
        "--rl-volume-slack-lower",
        type=float,
        default=0.05,
        help="Lower slack for volume-aware action masking (allowed volume >= target - slack).",
    )
    parser.add_argument(
        "--rl-volume-slack-upper",
        type=float,
        default=0.05,
        help="Upper slack for volume-aware action masking (allowed volume <= target + slack).",
    )
    parser.add_argument(
        "--rl-skip-threshold",
        type=float,
        default=0.95,
        help=(
            "Cosine-similarity threshold for smart skipping of terminal FEA. "
            "Set to 0 to disable smart skipping."
        ),
    )
    parser.add_argument(
        "--rl-skip-warmup-fraction",
        type=float,
        default=0.3,
        help="Fraction of total timesteps before smart skipping activates.",
    )
    parser.add_argument(
        "--rl-harmonic-clamp",
        type=float,
        default=10.0,
        help="Clamp for 1/compliance and 1/(1+penalty) terms in the harmonic-mean reward.",
    )
    parser.add_argument(
        "--rl-infeasible-terminal-reward",
        type=float,
        default=-1.0,
        help=(
            "Maximum negative reward when terminal mask fails feasibility filters. "
            "The actual reward is scaled by a soft gap derived from volume, islands, "
            "and support/load contact."
        ),
    )
    parser.add_argument(
        "--rl-ent-coef",
        type=float,
        default=0.03,
        help="Entropy bonus coefficient passed to MaskablePPO. Prevents entropy collapse on sparse rewards.",
    )
    parser.add_argument(
        "--rl-target-kl",
        type=float,
        default=0.03,
        help="Target KL divergence threshold for MaskablePPO. Set to 0 to disable.",
    )
    parser.add_argument(
        "--rl-best-harvest-topk",
        type=int,
        default=4,
        help=(
            "Top-K feasible candidates per rollout to re-evaluate in the main evaluator "
            "during training. Set to 0 to disable training-best harvesting."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=256,
        help="Maximum steps per RL episode. Scale with resolution for sparse reward.",
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


def _git_version_info() -> dict[str, Any]:
    def _run(args: list[str], cwd: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(cwd),
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    info: dict[str, Any] = {
        "branch": None,
        "commit": None,
        "commit_short": None,
        "dirty": None,
        "commit_subject": None,
        "commit_time": None,
    }
    try:
        override_commit = os.environ.get("GEN_RL_GIT_COMMIT")
        override_branch = os.environ.get("GEN_RL_GIT_BRANCH")
        start_dir = Path(__file__).resolve().parent
        toplevel = _run(["rev-parse", "--show-toplevel"], start_dir)
        cwd = Path(toplevel) if toplevel else start_dir
        commit = override_commit or _run(["rev-parse", "HEAD"], cwd)
        if commit:
            info["commit"] = commit
            info["commit_short"] = commit[:12]
        branch = override_branch or _run(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
        if branch and branch != "HEAD":
            info["branch"] = branch
        else:
            describe = _run(["describe", "--all", "--always", "HEAD"], cwd)
            if describe:
                info["branch"] = describe
        status = _run(["status", "--porcelain"], cwd)
        if status is not None:
            info["dirty"] = bool(status.strip())
        subject = _run(["log", "-1", "--pretty=%s"], cwd)
        if subject:
            info["commit_subject"] = subject
        commit_time = _run(["log", "-1", "--pretty=%cI"], cwd)
        if commit_time:
            info["commit_time"] = commit_time
    except Exception:
        return info
    return info


def _summary_payload(config: ProblemConfig, artifacts: Any) -> dict[str, Any]:
    return {
        "git": _git_version_info(),
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
            "rl_boundary_depth": config.rl_boundary_depth,
            "rl_stress_metric": config.rl_stress_metric,
            "rl_stress_hotspot_quantile": config.rl_stress_hotspot_quantile,
            "rl_stress_hotspot_dilate": config.rl_stress_hotspot_dilate,
            "rl_stop_penalty": config.rl_stop_penalty,
            "rl_degenerate_episode_window": config.rl_degenerate_episode_window,
            "rl_sparse_reward": config.rl_sparse_reward,
            "rl_n_envs": config.rl_n_envs,
            "rl_inference_rollouts": config.rl_inference_rollouts,
            "rl_policy_size": config.rl_policy_size,
            "rl_volume_slack_lower": config.rl_volume_slack_lower,
            "rl_volume_slack_upper": config.rl_volume_slack_upper,
            "rl_skip_threshold": config.rl_skip_threshold,
            "rl_skip_warmup_fraction": config.rl_skip_warmup_fraction,
            "rl_harmonic_clamp": config.rl_harmonic_clamp,
            "rl_infeasible_terminal_reward": config.rl_infeasible_terminal_reward,
            "rl_ent_coef": config.rl_ent_coef,
            "rl_target_kl": config.rl_target_kl,
            "rl_best_harvest_topk": config.rl_best_harvest_topk,
            "max_episode_steps": config.max_episode_steps,
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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _save_outputs(output_dir: Path, artifacts: Any, config: ProblemConfig) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    if config.pipeline_mode in ("direct64_exact", "rl_only_exact"):
        seed_name = "seed64.npy" if config.pipeline_mode == "direct64_exact" else f"seed{config.resolution}.npy"
        best_name = "best64.npy" if config.pipeline_mode == "direct64_exact" else f"best{config.resolution}.npy"
        png_name = "best64.png" if config.pipeline_mode == "direct64_exact" else f"best{config.resolution}.png"
        seed_key = "seed64" if config.pipeline_mode == "direct64_exact" else "seed"
        best_key = "best64" if config.pipeline_mode == "direct64_exact" else "best"
        png_key = "best64_png" if config.pipeline_mode == "direct64_exact" else "best_png"
        seed_path = output_dir / seed_name
        best_path = output_dir / best_name
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
        archive_path.write_text(
            json.dumps(
                _json_safe({"entries": archive_entries, "search_trace": artifacts.search_trace}),
                indent=2,
            )
        )
        saved[seed_key] = str(seed_path)
        saved[best_key] = str(best_path)
        saved["archive"] = str(archive_path)

        final_png = output_dir / png_name
        figure = plt.figure(dpi=100)
        print("\nFinal Cantilever beam design:")
        yellow_material = ListedColormap(["#ffffff", "#ffd84d"])
        plt.imshow(artifacts.best64, cmap=yellow_material, vmin=0, vmax=1)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(final_png, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        saved[png_key] = str(final_png)
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
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2))
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
        rl_boundary_depth=args.rl_boundary_depth,
        rl_stress_hotspot_quantile=args.rl_stress_hotspot_quantile,
        rl_stress_hotspot_dilate=args.rl_stress_hotspot_dilate,
        rl_stop_penalty=args.rl_stop_penalty,
        rl_degenerate_episode_window=args.rl_degenerate_episode_window,
        rl_sparse_reward=args.rl_sparse_reward,
        rl_n_envs=args.rl_n_envs,
        rl_inference_rollouts=args.rl_inference_rollouts,
        rl_policy_size=args.rl_policy_size,
        rl_volume_slack_lower=args.rl_volume_slack_lower,
        rl_volume_slack_upper=args.rl_volume_slack_upper,
        rl_skip_threshold=args.rl_skip_threshold,
        rl_skip_warmup_fraction=args.rl_skip_warmup_fraction,
        rl_harmonic_clamp=args.rl_harmonic_clamp,
        rl_infeasible_terminal_reward=args.rl_infeasible_terminal_reward,
        rl_ent_coef=args.rl_ent_coef,
        rl_target_kl=args.rl_target_kl,
        rl_best_harvest_topk=args.rl_best_harvest_topk,
        max_episode_steps=args.max_episode_steps,
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

    git_info = _git_version_info()
    if git_info.get("commit"):
        _progress_logger(
            "git version: "
            f"branch={git_info.get('branch') or '<detached>'}, "
            f"commit={git_info.get('commit_short')} "
            f"(dirty={git_info.get('dirty')}) "
            f"subject={git_info.get('commit_subject') or '-'}"
        )
    else:
        _progress_logger("git version: unavailable (not a git checkout or git not on PATH)")
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
