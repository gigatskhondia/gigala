from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .coarse_search import SearchCandidate, boundary_local_search, run_coarse_search
from .direct_search import DirectSearchArtifacts, run_direct_search_core
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


@dataclass
class DirectRLDegeneracyMonitor:
    episode_window: int
    immediate_stop_flags: list[bool] = field(default_factory=list)
    episodes_observed: int = 0
    stopped_early: bool = False
    reason: str | None = None

    def observe_episode(self, info: dict[str, Any]) -> bool:
        episode = info.get("episode")
        if not isinstance(episode, dict):
            return False
        self.episodes_observed += 1
        diagnostics = info.get("rl_diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        immediate_stop = (
            int(episode.get("l", 0)) <= 1
            and bool(diagnostics.get("stop_used"))
            and bool(info.get("stop_penalty_applied"))
            and int(diagnostics.get("accepted_removals", 0)) == 0
        )
        self.immediate_stop_flags.append(immediate_stop)
        if self.episode_window > 0 and len(self.immediate_stop_flags) > self.episode_window:
            self.immediate_stop_flags = self.immediate_stop_flags[-self.episode_window :]
        if (
            self.episode_window > 0
            and len(self.immediate_stop_flags) >= self.episode_window
            and all(self.immediate_stop_flags[-self.episode_window :])
        ):
            self.stopped_early = True
            self.reason = f"degenerate_immediate_stop_policy:{self.episode_window}_episodes"
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        recent_window = self.immediate_stop_flags[-self.episode_window :] if self.episode_window > 0 else []
        return {
            "episode_window": int(self.episode_window),
            "episodes_observed": int(self.episodes_observed),
            "immediate_stop_episodes_in_window": int(sum(recent_window)),
            "stopped_early": bool(self.stopped_early),
            "reason": self.reason,
        }


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


def _evaluation_ranking_key(result: EvalResult) -> tuple[float, float, float, int]:
    feasible_rank = 0.0 if result.passed_filters else 1.0
    return (
        feasible_rank,
        float(result.score),
        float(result.compliance),
        int(result.smoothness),
    )


def _select_monotonic_refinement(
    incumbent_mask: np.ndarray,
    incumbent_eval: EvalResult,
    candidate_mask: np.ndarray,
    candidate_eval: EvalResult,
) -> tuple[np.ndarray, EvalResult, dict[str, Any]]:
    incumbent_key = _evaluation_ranking_key(incumbent_eval)
    candidate_key = _evaluation_ranking_key(candidate_eval)
    accepted = candidate_key < incumbent_key
    if accepted:
        reason = "accepted_better_rl_candidate"
        selected_mask = candidate_mask
        selected_eval = candidate_eval
    elif not candidate_eval.passed_filters:
        reason = f"rejected_invalid_rl_candidate:{candidate_eval.invalid_reason or 'unknown'}"
        selected_mask = incumbent_mask
        selected_eval = incumbent_eval
    else:
        reason = "rejected_non_improving_rl_candidate"
        selected_mask = incumbent_mask
        selected_eval = incumbent_eval
    return selected_mask, selected_eval, {
        "accepted": accepted,
        "reason": reason,
        "incumbent_score": float(incumbent_eval.score),
        "candidate_score": float(candidate_eval.score),
        "selected_score": float(selected_eval.score),
        "candidate_valid": bool(candidate_eval.passed_filters),
        "incumbent_valid": bool(incumbent_eval.passed_filters),
    }


def _is_better_result(candidate: EvalResult, incumbent: EvalResult) -> bool:
    return _evaluation_ranking_key(candidate) < _evaluation_ranking_key(incumbent)


def _build_rl_seed_list(best64: np.ndarray, archive_best: list[np.ndarray], top_k: int) -> list[np.ndarray]:
    seeds: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in [best64, *archive_best]:
        signature = np.ascontiguousarray(mask).tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        seeds.append(mask.copy())
        if len(seeds) >= top_k:
            break
    return seeds


def _maybe_run_rl(
    seed_mask: np.ndarray,
    config: ProblemConfig,
    evaluator: Evaluator,
    warnings: list[str],
    seed_evaluation: EvalResult | None = None,
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
    if seed_evaluation is not None and hasattr(base_env, "register_seed_evaluation"):
        base_env.register_seed_evaluation(seed_evaluation)
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
    rollout_mask = base_env.render()
    rollout_eval = evaluator.evaluate(rollout_mask, "full64")
    selected_mask = rollout_mask.copy()
    selected_source = "deterministic_rollout"
    if hasattr(base_env, "best_result"):
        best_mask, best_eval = base_env.best_result()
        if best_eval is not None and _is_better_result(best_eval, rollout_eval):
            selected_mask = best_mask
            selected_source = "training_best"
    if progress:
        progress(
            f"RL inference rollout completed; full64_evals={int(evaluator.fea_counts['full64'])}, "
            f"selected_source={selected_source}"
        )
    return selected_mask


def _maybe_run_direct_rl(
    seed_mask: np.ndarray,
    config: ProblemConfig,
    evaluator: Evaluator,
    warnings: list[str],
    progress: ProgressFn | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not config.enable_rl:
        warnings.append("RL refinement disabled by config; returning direct-search best seed.")
        if progress:
            progress("RL refinement disabled by config; returning direct-search best seed.")
        return seed_mask, {"skipped": True, "reason": "rl_disabled"}
    try:  # pragma: no cover - optional dependency
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        from stable_baselines3.common.callbacks import BaseCallback
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.monitor import Monitor
    except Exception:
        warnings.append("MaskablePPO dependencies are unavailable; returning direct-search best seed.")
        if progress:
            progress("MaskablePPO dependencies are unavailable; returning direct-search best seed.")
        return seed_mask, {"skipped": True, "reason": "missing_maskableppo_dependencies"}

    from .refine_env import default_policy_kwargs, make_direct64_refine_env

    device = _resolve_rl_device(config, progress=progress, torch_module=torch)
    if progress:
        progress(
            f"starting direct64 RL refinement: total_timesteps={config.rl_total_timesteps}, "
            f"max_rl_full_evals={config.max_rl_full_evals}, device={device}"
        )
    env = make_direct64_refine_env(seed_mask, config, evaluator=evaluator)
    monitored = Monitor(env)
    base_env = monitored.unwrapped
    degeneracy_monitor = DirectRLDegeneracyMonitor(episode_window=config.rl_degenerate_episode_window)

    class _DirectRLEarlyExitCallback(BaseCallback):
        def _on_step(self) -> bool:
            infos = self.locals.get("infos")
            if not isinstance(infos, list):
                return True
            for info in infos:
                if isinstance(info, dict) and degeneracy_monitor.observe_episode(info):
                    if progress:
                        progress(
                            "direct64 RL early exit triggered: "
                            f"{degeneracy_monitor.reason or 'degenerate_policy_detected'}"
                        )
                    return False
            return True

    callback = _DirectRLEarlyExitCallback()
    model = MaskablePPO(
        "CnnPolicy",
        monitored,
        policy_kwargs=default_policy_kwargs(),
        verbose=1 if progress else 0,
        seed=config.random_seed,
        device=device,
    )
    model.learn(total_timesteps=config.rl_total_timesteps, callback=callback)
    if degeneracy_monitor.stopped_early and degeneracy_monitor.reason:
        warnings.append(f"direct64 RL early exit: {degeneracy_monitor.reason}")
    if progress:
        progress("direct64 RL training completed; running deterministic inference rollout.")

    observation, _ = monitored.reset()
    done = False
    last_info: dict[str, Any] = {}
    while not done:
        action_masks = base_env.action_masks()
        action, _ = model.predict(observation, action_masks=action_masks, deterministic=True)
        observation, _reward, terminated, truncated, last_info = monitored.step(action)
        done = bool(terminated or truncated)
    diagnostics = base_env.rl_diagnostics() if hasattr(base_env, "rl_diagnostics") else {}
    diagnostics = {
        **diagnostics,
        "device": device,
        "rl_total_timesteps": int(config.rl_total_timesteps),
        "training_monitor": degeneracy_monitor.snapshot(),
        "last_info": last_info,
    }
    if progress:
        progress(
            "direct64 RL inference rollout completed; "
            f"full64_evals={int(evaluator.fea_counts['full64'])}, "
            f"boundary_candidates={diagnostics.get('boundary_candidate_count', 0)}, "
            f"hotspot_candidates={diagnostics.get('hotspot_candidate_count', 0)}, "
            f"union_candidates={diagnostics.get('union_candidate_count', 0)}, "
            f"accepted_removals={diagnostics.get('accepted_removals', 0)}, "
            f"rejected_invalid={diagnostics.get('rejected_invalid_removals', 0)}, "
            f"rejected_non_improving={diagnostics.get('rejected_non_improving_removals', 0)}"
        )
    return base_env.render(), diagnostics


def run_multistage_search(config: ProblemConfig, *, progress: ProgressFn | None = None) -> StageArtifacts:
    start = time.time()
    deadline = start + config.runtime_budget_hours * 3600.0 if config.has_runtime_budget() else None
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
#     if config.resolution >= 128:
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
        stage64_seed_eval = stage64_candidates[0].evaluation
        if progress:
            progress(
                f"stage {config.resolution} seed completed: best_score={stage64_seed_eval.score:.4f}, "
                f"full64_evals={int(evaluator.fea_counts['full64'])}"
            )
        rl_candidate = _maybe_run_rl(stage64_seed, config, evaluator, warnings, seed_evaluation=stage64_seed_eval, progress=progress)
        rl_candidate_eval = evaluator.evaluate(rl_candidate, "full64")
        refined64, refined64_eval, selection = _select_monotonic_refinement(
            stage64_seed,
            stage64_seed_eval,
            rl_candidate,
            rl_candidate_eval,
        )
        metrics["refined64_seed"] = _result_to_dict(stage64_seed_eval)
        metrics["refined64_rl_candidate"] = _result_to_dict(rl_candidate_eval)
        metrics["refined64_selection"] = selection
        metrics["refined64"] = _result_to_dict(refined64_eval)
        if progress:
            progress(
                f"final refined64 evaluated: score={metrics['refined64']['score']:.4f}, "
                f"accepted_rl={selection['accepted']}, "
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


def run_direct64_exact_search(config: ProblemConfig, *, progress: ProgressFn | None = None) -> DirectSearchArtifacts:
    direct_config = dataclasses.replace(config, pipeline_mode="direct64_exact")
    artifacts = run_direct_search_core(direct_config, progress=progress)
    if not direct_config.enable_rl:
        artifacts.metrics["final64"] = dict(artifacts.metrics["best64"])
        artifacts.metrics["rl_refined64_diagnostics"] = {}
        artifacts.metrics["final64_diagnostics"] = {}
        artifacts.metrics["rl_selection"] = {
            "accepted": False,
            "reason": "rl_disabled",
            "incumbent_score": float(artifacts.metrics["best64"]["score"]),
            "candidate_score": float(artifacts.metrics["best64"]["score"]),
            "selected_score": float(artifacts.metrics["best64"]["score"]),
            "candidate_valid": bool(artifacts.metrics["best64"]["passed_filters"]),
            "incumbent_valid": bool(artifacts.metrics["best64"]["passed_filters"]),
            "diagnostics": {},
        }
        artifacts.metrics["rl_trials"] = []
        return artifacts

    rl_start = time.time()
    rl_config = dataclasses.replace(direct_config, max_full_evals=max(direct_config.max_rl_full_evals, 1_000_000))
    evaluator = Evaluator(rl_config)
    warnings = list(artifacts.warnings)
    global_best_mask = artifacts.best64.copy()
    global_best_eval = evaluator.evaluate(global_best_mask, "full64")
    trial_results: list[dict[str, Any]] = []
    best_rl_candidate_mask = global_best_mask.copy()
    best_rl_candidate_eval: EvalResult | None = None

    rl_seeds = _build_rl_seed_list(artifacts.best64, artifacts.archive_best, direct_config.rl_archive_top_k)
    if progress:
        progress(
            f"starting direct64 multi-start RL refinement from {len(rl_seeds)} archive seeds "
            f"(requested_top_k={direct_config.rl_archive_top_k})"
        )

    overall_selection = {
        "accepted": False,
        "reason": "no_improving_rl_candidate",
        "incumbent_score": float(global_best_eval.score),
        "candidate_score": float(global_best_eval.score),
        "selected_score": float(global_best_eval.score),
        "candidate_valid": bool(global_best_eval.passed_filters),
        "incumbent_valid": bool(global_best_eval.passed_filters),
        "seed_index": 0,
        "seed_count": len(rl_seeds),
        "diagnostics": {},
    }

    for seed_index, seed_mask in enumerate(rl_seeds, start=1):
        trial_config = dataclasses.replace(rl_config, random_seed=direct_config.random_seed + seed_index - 1)
        seed_eval = evaluator.evaluate(seed_mask, "full64")
        if progress:
            progress(
                f"direct64 RL trial {seed_index}/{len(rl_seeds)}: seed_score={seed_eval.score:.4f}, "
                f"seed_valid={seed_eval.passed_filters}"
            )
        rl_candidate, rl_diagnostics = _maybe_run_direct_rl(seed_mask, trial_config, evaluator, warnings, progress=progress)
        rl_eval = evaluator.evaluate(rl_candidate, "full64")
        selected_mask, selected_eval, selection = _select_monotonic_refinement(
            seed_mask,
            seed_eval,
            rl_candidate,
            rl_eval,
        )
        selection["seed_index"] = seed_index
        selection["seed_count"] = len(rl_seeds)
        trial_results.append(
            {
                "seed_index": seed_index,
                "seed_score": float(seed_eval.score),
                "seed_valid": bool(seed_eval.passed_filters),
                "candidate": _result_to_dict(rl_eval),
                "selection": selection,
                "selected": _result_to_dict(selected_eval),
                "diagnostics": rl_diagnostics,
            }
        )
        if best_rl_candidate_eval is None or _is_better_result(rl_eval, best_rl_candidate_eval):
            best_rl_candidate_mask = rl_candidate.copy()
            best_rl_candidate_eval = rl_eval
        if _is_better_result(selected_eval, global_best_eval):
            global_best_mask = selected_mask.copy()
            global_best_eval = selected_eval
            overall_selection = {**selection, "diagnostics": rl_diagnostics}
        artifacts.search_trace.append(
            {
                "event": "rl_refinement_trial",
                "seed_index": seed_index,
                "seed_score": float(seed_eval.score),
                "candidate_score": float(rl_eval.score),
                "accepted": bool(selection["accepted"]),
                "selected_score": float(selected_eval.score),
                "diagnostics": rl_diagnostics,
            }
        )
        if progress:
            progress(
                f"direct64 RL trial {seed_index}/{len(rl_seeds)} selection: accepted={selection['accepted']}, "
                f"reason={selection['reason']}, selected_score={selected_eval.score:.4f}, "
                f"accepted_removals={rl_diagnostics.get('accepted_removals', 0)}, "
                f"boundary_candidates={rl_diagnostics.get('boundary_candidate_count', 0)}, "
                f"hotspot_candidates={rl_diagnostics.get('hotspot_candidate_count', 0)}"
            )

    artifacts.best64 = global_best_mask
    artifacts.metrics["rl_refined64"] = _result_to_dict(best_rl_candidate_eval or global_best_eval)
    best_trial_diagnostics = {}
    if trial_results:
        best_trial_diagnostics = min(
            trial_results,
            key=lambda trial: (
                0.0 if trial["candidate"]["passed_filters"] else 1.0,
                float(trial["candidate"]["score"]),
            ),
        )["diagnostics"]
    artifacts.metrics["rl_refined64_diagnostics"] = best_trial_diagnostics
    artifacts.metrics["rl_trials"] = trial_results
    artifacts.metrics["rl_selection"] = overall_selection
    artifacts.metrics["final64"] = _result_to_dict(global_best_eval)
    artifacts.metrics["final64_diagnostics"] = dict(best_trial_diagnostics if overall_selection["accepted"] else {})
    rl_counts = evaluation_snapshot(evaluator)
    artifacts.fea_counts = {
        "proxy16": float(artifacts.fea_counts.get("proxy16", 0.0) + rl_counts.get("proxy16", 0.0)),
        "proxy32": float(artifacts.fea_counts.get("proxy32", 0.0) + rl_counts.get("proxy32", 0.0)),
        "full64": float(artifacts.fea_counts.get("full64", 0.0) + rl_counts.get("full64", 0.0)),
        "cache_hits": float(artifacts.fea_counts.get("cache_hits", 0.0) + rl_counts.get("cache_hits", 0.0)),
        "cache_size": float(max(artifacts.fea_counts.get("cache_size", 0.0), rl_counts.get("cache_size", 0.0))),
    }
    artifacts.warnings = warnings
    artifacts.runtime += time.time() - rl_start
    artifacts.search_trace.append(
        {
            "event": "rl_refinement",
            "full64_evals": int(artifacts.fea_counts["full64"]),
            "seed_count": len(rl_seeds),
            "accepted": bool(overall_selection["accepted"]),
            "best_score": float(global_best_eval.score),
            "best_volume": float(global_best_eval.volume_fraction),
            "best_islands": int(global_best_eval.islands),
            "candidate_score": float((best_rl_candidate_eval or global_best_eval).score),
        }
    )
    if progress:
        progress(
            f"direct64 RL selection: accepted={overall_selection['accepted']}, "
            f"reason={overall_selection['reason']}, final_score={global_best_eval.score:.4f}"
        )
    return artifacts


def run_rl_only_exact_search(config: ProblemConfig, *, progress: ProgressFn | None = None) -> DirectSearchArtifacts:
    rl_only_config = dataclasses.replace(
        config,
        pipeline_mode="rl_only_exact",
        direct_population=max(1, config.direct_population),
    )
    start = time.time()
    evaluator = Evaluator(rl_only_config)
    warnings: list[str] = []
    seed_mask = np.ones((rl_only_config.resolution, rl_only_config.resolution), dtype=np.uint8)
    seed_eval = evaluator.evaluate(seed_mask, "full64")
    search_trace: list[dict[str, Any]] = [
        {
            "event": "rl_only_seed",
            "seed_type": "full_solid",
            "score": float(seed_eval.score),
            "volume": float(seed_eval.volume_fraction),
            "passed_filters": bool(seed_eval.passed_filters),
            "invalid_reason": seed_eval.invalid_reason,
        }
    ]
    if progress:
        progress(
            f"rl-only exact search started: resolution={rl_only_config.resolution}, "
            f"enable_rl={rl_only_config.enable_rl}, seed_score={seed_eval.score:.4f}"
        )

    if rl_only_config.enable_rl:
        rl_candidate = _maybe_run_rl(
            seed_mask,
            rl_only_config,
            evaluator,
            warnings,
            seed_evaluation=seed_eval,
            progress=progress,
        )
        rl_candidate_eval = evaluator.evaluate(rl_candidate, "full64")
    else:
        warnings.append("RL-only pipeline launched with RL disabled; returning full-solid seed.")
        if progress:
            progress("RL-only pipeline launched with RL disabled; returning full-solid seed.")
        rl_candidate = seed_mask.copy()
        rl_candidate_eval = seed_eval

    final_mask, final_eval, selection = _select_monotonic_refinement(
        seed_mask,
        seed_eval,
        rl_candidate,
        rl_candidate_eval,
    )
    runtime = time.time() - start
    search_trace.append(
        {
            "event": "rl_only_refinement",
            "seed_score": float(seed_eval.score),
            "candidate_score": float(rl_candidate_eval.score),
            "accepted": bool(selection["accepted"]),
            "final_score": float(final_eval.score),
        }
    )
    metrics: dict[str, dict[str, Any]] = {
        "seed": _result_to_dict(seed_eval),
        "rl_candidate": _result_to_dict(rl_candidate_eval),
        "rl_selection": selection,
        "final": _result_to_dict(final_eval),
    }
    if progress:
        progress(
            f"rl-only exact search finished: runtime_sec={runtime:.2f}, "
            f"accepted_rl={selection['accepted']}, final_score={final_eval.score:.4f}, "
            f"full64_evals={int(evaluator.fea_counts['full64'])}, cache_hits={evaluator.cache_hits}"
        )
    return DirectSearchArtifacts(
        initial_population=[seed_mask.copy()],
        archive_best=[seed_mask.copy()],
        best64=final_mask.copy(),
        metrics=metrics,
        fea_counts=evaluation_snapshot(evaluator),
        runtime=runtime,
        warnings=warnings,
        search_trace=search_trace,
    )


def run_search(config: ProblemConfig, *, progress: ProgressFn | None = None) -> StageArtifacts | DirectSearchArtifacts:
    if config.pipeline_mode == "direct64_exact":
        return run_direct64_exact_search(config, progress=progress)
    if config.pipeline_mode == "rl_only_exact":
        return run_rl_only_exact_search(config, progress=progress)
    return run_multistage_search(config, progress=progress)
