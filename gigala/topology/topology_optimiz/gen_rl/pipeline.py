from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .coarse_search import SearchCandidate, boundary_local_search, run_coarse_search
from .fem import EvalResult, Evaluator, ProblemConfig, evaluation_snapshot
from .representation import infer_stage_resolutions, upsample_binary_mask


@dataclass
class StageArtifacts:
    coarse16: np.ndarray
    refined32: np.ndarray | None
    refined64: np.ndarray | None
    metrics: dict[str, dict[str, Any]]
    fea_counts: dict[str, float]
    runtime: float
    warnings: list[str] = field(default_factory=list)


ProgressFn = Callable[[str], None]


def _result_to_dict(result: EvalResult) -> dict[str, Any]:
    return dataclasses.asdict(result)


def _resolve_rl_device(config: ProblemConfig, *, progress: ProgressFn | None = None, torch_module: Any | None = None) -> str:
    requested = config.rl_device
    if torch_module is None:
        try:  # pragma: no cover - optional dependency
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:
            if progress:
                progress("torch is unavailable; RL device forced to cpu.")
            return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if bool(torch_module.cuda.is_available()):
            return "cuda"
        if progress:
            progress("requested rl_device=cuda but CUDA is unavailable; falling back to cpu.")
        return "cpu"

    if requested == "mps":
        if hasattr(torch_module.backends, "mps") and bool(torch_module.backends.mps.is_available()):
            return "mps"
        if progress:
            progress("requested rl_device=mps but MPS is unavailable; falling back to cpu.")
        return "cpu"

    if requested == "auto":
        if hasattr(torch_module.backends, "mps") and bool(torch_module.backends.mps.is_available()):
            return "mps"
        if bool(torch_module.cuda.is_available()):
            return "cuda"
        return "cpu"

    if progress:
        progress(f"unknown rl_device={requested}; falling back to cpu.")
    return "cpu"


def evaluate(
    mask: np.ndarray,
    fidelity: str,
    *,
    config: ProblemConfig | None = None,
    evaluator: Evaluator | None = None,
) -> EvalResult:
    if evaluator is None:
        if config is None:
            raise ValueError("Either config or evaluator must be provided.")
        evaluator = Evaluator(config)
    return evaluator.evaluate(mask, fidelity)  # type: ignore[arg-type]


def _maybe_run_rl(
    seed_mask: np.ndarray,
    config: ProblemConfig,
    evaluator: Evaluator,
    warnings: list[str],
    progress: ProgressFn | None = None,
) -> np.ndarray:
    if not config.enable_rl:
        warnings.append("RL refinement disabled by config; returning boundary-refined seed.")
        if progress:
            progress("RL refinement disabled by config; returning boundary-refined seed.")
        return seed_mask
    try:  # pragma: no cover - optional dependency
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.monitor import Monitor
    except Exception:
        warnings.append("MaskablePPO dependencies are unavailable; returning boundary-refined seed.")
        if progress:
            progress("MaskablePPO dependencies are unavailable; returning boundary-refined seed.")
        return seed_mask

    from .refine_env import default_policy_kwargs, make_refine_env

    device = _resolve_rl_device(config, progress=progress, torch_module=torch)
    if progress:
        progress(
            f"starting RL refinement: total_timesteps={config.rl_total_timesteps}, "
            f"max_rl_full_evals={config.max_rl_full_evals}, device={device}"
        )
    env = make_refine_env(seed_mask, config, evaluator=evaluator)
    monitored = Monitor(env)
    base_env = monitored.unwrapped
    model = MaskablePPO(
        "CnnPolicy",
        monitored,
        policy_kwargs=default_policy_kwargs(),
        verbose=1 if progress else 0,
        seed=config.random_seed,
        device=device,
    )
    model.learn(total_timesteps=config.rl_total_timesteps)
    if progress:
        progress("RL training completed; running deterministic inference rollout.")

    observation, _ = monitored.reset()
    done = False
    while not done:
        action_masks = base_env.action_masks()
        action, _ = model.predict(observation, action_masks=action_masks, deterministic=True)
        observation, _reward, terminated, truncated, _info = monitored.step(action)
        done = bool(terminated or truncated)
    if progress:
        progress(f"RL inference rollout completed; full64_evals={int(evaluator.fea_counts['full64'])}")
    return base_env.render()


def run_multistage_search(config: ProblemConfig, *, progress: ProgressFn | None = None) -> StageArtifacts:
    start = time.time()
    deadline = start + config.runtime_budget_hours * 3600.0
    evaluator = Evaluator(config)
    warnings: list[str] = []
    stage_resolutions = infer_stage_resolutions(config.resolution)
    if progress:
        progress(
            f"pipeline started: final_resolution={config.resolution}, stages={stage_resolutions}, "
            f"runtime_budget_hours={config.runtime_budget_hours}, enable_rl={config.enable_rl}, "
            f"rl_device={config.rl_device}"
        )

    coarse_candidates = run_coarse_search(
        evaluator,
        config,
        resolution=stage_resolutions[0],
        top_k=max(config.stage32_top_k, 8),
        deadline=deadline,
        progress=progress,
    )
    if not coarse_candidates:
        raise RuntimeError("Coarse search did not produce any valid candidates.")

    coarse16 = coarse_candidates[0].mask
    metrics: dict[str, dict[str, Any]] = {
        "coarse16": _result_to_dict(coarse_candidates[0].evaluation),
    }
    if progress:
        progress(
            f"coarse16 completed: best_score={coarse_candidates[0].evaluation.score:.4f}, "
            f"proxy16_evals={int(evaluator.fea_counts['proxy16'])}"
        )

    refined32: np.ndarray | None = None
    refined64: np.ndarray | None = None

    if len(stage_resolutions) >= 2:
        second_resolution = stage_resolutions[1]
        if progress:
            progress(
                f"starting stage {second_resolution} boundary refinement with "
                f"{min(len(coarse_candidates), config.stage32_top_k)} promoted seeds"
            )
        seeds32 = [upsample_binary_mask(candidate.mask, second_resolution) for candidate in coarse_candidates[: config.stage32_top_k]]
        stage32_candidates = boundary_local_search(
            seeds32,
            evaluator,
            config,
            resolution=second_resolution,
            fidelity="proxy32" if second_resolution >= 32 else "proxy16",
            patch_sizes=(2,),
            top_k=max(config.stage64_top_k, 2),
            steps=config.local_search_steps32,
            deadline=deadline,
            progress=progress,
        )
        refined32 = stage32_candidates[0].mask
        metrics["refined32"] = _result_to_dict(stage32_candidates[0].evaluation)
        if progress:
            progress(
                f"stage {second_resolution} completed: best_score={stage32_candidates[0].evaluation.score:.4f}, "
                f"proxy32_evals={int(evaluator.fea_counts['proxy32'])}"
            )
    else:
        stage32_candidates = coarse_candidates[: config.stage64_top_k]

    if config.resolution >= 64:
        seeds64_source = stage32_candidates[: config.stage64_top_k]
        if progress:
            progress(
                f"starting stage {config.resolution} boundary refinement with "
                f"{len(seeds64_source)} promoted seeds"
            )
        seeds64 = [upsample_binary_mask(candidate.mask, config.resolution) for candidate in seeds64_source]
        stage64_candidates = boundary_local_search(
            seeds64,
            evaluator,
            config,
            resolution=config.resolution,
            fidelity="full64",
            patch_sizes=(2, 1),
            top_k=1,
            steps=config.local_search_steps64,
            deadline=deadline,
            progress=progress,
        )
        stage64_seed = stage64_candidates[0].mask
        if progress:
            progress(
                f"stage {config.resolution} seed completed: best_score={stage64_candidates[0].evaluation.score:.4f}, "
                f"full64_evals={int(evaluator.fea_counts['full64'])}"
            )
        refined64 = _maybe_run_rl(stage64_seed, config, evaluator, warnings, progress=progress)
        metrics["refined64_seed"] = _result_to_dict(stage64_candidates[0].evaluation)
        metrics["refined64"] = _result_to_dict(evaluator.evaluate(refined64, "full64"))
        if progress:
            progress(
                f"final refined64 evaluated: score={metrics['refined64']['score']:.4f}, "
                f"full64_evals={int(evaluator.fea_counts['full64'])}, cache_hits={evaluator.cache_hits}"
            )
    elif refined32 is not None:
        refined64 = refined32

    runtime = time.time() - start
    if progress:
        progress(
            f"pipeline finished: runtime_sec={runtime:.2f}, "
            f"proxy16_evals={int(evaluator.fea_counts['proxy16'])}, "
            f"proxy32_evals={int(evaluator.fea_counts['proxy32'])}, "
            f"full64_evals={int(evaluator.fea_counts['full64'])}, cache_hits={evaluator.cache_hits}"
        )
    return StageArtifacts(
        coarse16=coarse16,
        refined32=refined32,
        refined64=refined64,
        metrics=metrics,
        fea_counts=evaluation_snapshot(evaluator),
        runtime=runtime,
        warnings=warnings,
    )
