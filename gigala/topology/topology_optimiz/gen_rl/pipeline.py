from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .coarse_search import SearchCandidate, boundary_local_search, run_coarse_search
from .fem import EvalResult, Evaluator, ProblemConfig, evaluation_snapshot
from .representation import infer_stage_resolutions, upsample_binary_mask


@dataclass(slots=True)
class StageArtifacts:
    coarse16: np.ndarray
    refined32: np.ndarray | None
    refined64: np.ndarray | None
    metrics: dict[str, dict[str, Any]]
    fea_counts: dict[str, float]
    runtime: float
    warnings: list[str] = field(default_factory=list)


def _result_to_dict(result: EvalResult) -> dict[str, Any]:
    return dataclasses.asdict(result)


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
) -> np.ndarray:
    if not config.enable_rl:
        warnings.append("RL refinement disabled by config; returning boundary-refined seed.")
        return seed_mask
    try:  # pragma: no cover - optional dependency
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.monitor import Monitor
    except Exception:
        warnings.append("MaskablePPO dependencies are unavailable; returning boundary-refined seed.")
        return seed_mask

    from .refine_env import default_policy_kwargs, make_refine_env

    env = make_refine_env(seed_mask, config, evaluator=evaluator)
    monitored = Monitor(env)
    base_env = monitored.unwrapped
    model = MaskablePPO(
        "CnnPolicy",
        monitored,
        policy_kwargs=default_policy_kwargs(),
        verbose=0,
        seed=config.random_seed,
    )
    model.learn(total_timesteps=config.rl_total_timesteps)

    observation, _ = monitored.reset()
    done = False
    while not done:
        action_masks = base_env.action_masks()
        action, _ = model.predict(observation, action_masks=action_masks, deterministic=True)
        observation, _reward, terminated, truncated, _info = monitored.step(action)
        done = bool(terminated or truncated)
    return base_env.render()


def run_multistage_search(config: ProblemConfig) -> StageArtifacts:
    start = time.time()
    deadline = start + config.runtime_budget_hours * 3600.0
    evaluator = Evaluator(config)
    warnings: list[str] = []
    stage_resolutions = infer_stage_resolutions(config.resolution)

    coarse_candidates = run_coarse_search(
        evaluator,
        config,
        resolution=stage_resolutions[0],
        top_k=max(config.stage32_top_k, 8),
        deadline=deadline,
    )
    if not coarse_candidates:
        raise RuntimeError("Coarse search did not produce any valid candidates.")

    coarse16 = coarse_candidates[0].mask
    metrics: dict[str, dict[str, Any]] = {
        "coarse16": _result_to_dict(coarse_candidates[0].evaluation),
    }

    refined32: np.ndarray | None = None
    refined64: np.ndarray | None = None

    if len(stage_resolutions) >= 2:
        second_resolution = stage_resolutions[1]
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
        )
        refined32 = stage32_candidates[0].mask
        metrics["refined32"] = _result_to_dict(stage32_candidates[0].evaluation)
    else:
        stage32_candidates = coarse_candidates[: config.stage64_top_k]

    if config.resolution >= 64:
        seeds64_source = stage32_candidates[: config.stage64_top_k]
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
        )
        stage64_seed = stage64_candidates[0].mask
        refined64 = _maybe_run_rl(stage64_seed, config, evaluator, warnings)
        metrics["refined64_seed"] = _result_to_dict(stage64_candidates[0].evaluation)
        metrics["refined64"] = _result_to_dict(evaluator.evaluate(refined64, "full64"))
    elif refined32 is not None:
        refined64 = refined32

    runtime = time.time() - start
    return StageArtifacts(
        coarse16=coarse16,
        refined32=refined32,
        refined64=refined64,
        metrics=metrics,
        fea_counts=evaluation_snapshot(evaluator),
        runtime=runtime,
        warnings=warnings,
    )
