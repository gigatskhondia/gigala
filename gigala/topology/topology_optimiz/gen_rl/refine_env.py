from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .fem import ElementFieldDiagnostics, Evaluator, ProblemConfig
from .metrics import (
    calculate_smoothness_metric,
    count_islands,
    ensure_binary,
    prune_isolated_cells,
    volume_fraction,
)
from .representation import frontier_band, stack_observation

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - optional dependency
    gym = None
    spaces = None


try:  # pragma: no cover - optional dependency
    import torch as th
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:  # pragma: no cover - optional dependency
    th = None
    BaseFeaturesExtractor = object  # type: ignore[assignment]


@dataclass(frozen=True)
class GridAction:
    kind: str
    row: int
    col: int
    size: int


@dataclass(frozen=True)
class DirectGridAction:
    kind: str
    row: int
    col: int
    size: int


def build_action_catalog(resolution: int) -> list[GridAction]:
    actions: list[GridAction] = []
    for row in range(resolution):
        for col in range(resolution):
            actions.append(GridAction("add_1x1", row, col, 1))
            actions.append(GridAction("remove_1x1", row, col, 1))
    for size in (2, 4):
        limit = resolution - size + 1
        for row in range(limit):
            for col in range(limit):
                actions.append(GridAction(f"remove_{size}x{size}", row, col, size))
    return actions


def compute_action_mask(mask: np.ndarray, catalog: list[GridAction], immutable_mask: np.ndarray, frontier_width: int) -> np.ndarray:
    binary = ensure_binary(mask)
    frontier = frontier_band(binary, width=frontier_width)
    valid = np.zeros(len(catalog), dtype=bool)
    for idx, action in enumerate(catalog):
        patch = binary[action.row : action.row + action.size, action.col : action.col + action.size]
        immutable_patch = immutable_mask[action.row : action.row + action.size, action.col : action.col + action.size]
        frontier_patch = frontier[action.row : action.row + action.size, action.col : action.col + action.size]
        if action.kind == "add_1x1":
            valid[idx] = patch.shape == (1, 1) and patch[0, 0] == 0 and frontier_patch.any()
        elif action.kind == "remove_1x1":
            valid[idx] = (
                patch.shape == (1, 1)
                and patch[0, 0] == 1
                and frontier_patch.any()
                and immutable_patch.sum() == 0
                and _count_neighbors(binary, action.row, action.col) >= 2
            )
        else:
            valid[idx] = patch.shape == (action.size, action.size) and patch.all() and frontier_patch.any() and immutable_patch.sum() == 0
    return valid


def _count_neighbors(mask: np.ndarray, row: int, col: int) -> int:
    row_min = max(0, row - 1)
    row_max = min(mask.shape[0], row + 2)
    col_min = max(0, col - 1)
    col_max = min(mask.shape[1], col + 2)
    patch = mask[row_min:row_max, col_min:col_max]
    return int(patch.sum() - mask[row, col])


def apply_action(mask: np.ndarray, action: GridAction) -> np.ndarray:
    binary = ensure_binary(mask).copy()
    if action.kind == "add_1x1":
        binary[action.row, action.col] = 1
    else:
        binary[action.row : action.row + action.size, action.col : action.col + action.size] = 0
    return prune_isolated_cells(binary)


def build_direct_action_catalog(resolution: int) -> list[DirectGridAction]:
    actions: list[DirectGridAction] = []
    for row in range(resolution):
        for col in range(resolution):
            actions.append(DirectGridAction("remove_cell", row, col, 1))
    actions.append(DirectGridAction("stop", 0, 0, 0))
    return actions


def compute_boundary_removal_mask(mask: np.ndarray, immutable_mask: np.ndarray, *, depth: int = 1) -> np.ndarray:
    binary = ensure_binary(mask)
    mutable = np.asarray(immutable_mask, dtype=np.uint8) == 0
    if depth < 1:
        return np.zeros_like(binary, dtype=bool)
    void_mask = binary == 0
    structure = ndimage.generate_binary_structure(2, 1)
    boundary = np.zeros_like(binary, dtype=bool)
    current = binary.astype(bool)
    for _ in range(depth):
        neighbor_voids = ndimage.convolve(void_mask.astype(np.uint8), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8), mode="constant", cval=1)
        layer = current & (neighbor_voids > 0)
        boundary |= layer
        current = ndimage.binary_erosion(current, structure=structure, iterations=1)
    return boundary & mutable & (binary == 1)


def compute_stress_hotspot_mask(
    mask: np.ndarray,
    immutable_mask: np.ndarray,
    von_mises: np.ndarray,
    *,
    quantile: float,
    dilate: int,
) -> np.ndarray:
    binary = ensure_binary(mask)
    mutable = np.asarray(immutable_mask, dtype=np.uint8) == 0
    solid_mutable = (binary == 1) & mutable
    hotspot = np.zeros_like(binary, dtype=bool)
    if not solid_mutable.any():
        return hotspot
    stress_field = np.asarray(von_mises, dtype=float)
    solid_indices = np.flatnonzero(solid_mutable.ravel())
    solid_stress = stress_field.ravel()[solid_indices]
    hotspot_count = max(1, int(np.ceil(solid_stress.size * max(1.0 - quantile, 0.0))))
    top_indices = solid_indices[np.argpartition(solid_stress, -hotspot_count)[-hotspot_count:]]
    hotspot.ravel()[top_indices] = True
    if dilate > 0:
        structure = ndimage.generate_binary_structure(2, 1)
        hotspot = ndimage.binary_dilation(hotspot, structure=structure, iterations=dilate)
    return hotspot & solid_mutable


def compute_direct_editable_masks(
    mask: np.ndarray,
    immutable_mask: np.ndarray,
    von_mises: np.ndarray,
    *,
    boundary_depth: int,
    hotspot_quantile: float,
    hotspot_dilate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary_mask = compute_boundary_removal_mask(mask, immutable_mask, depth=boundary_depth)
    hotspot_mask = compute_stress_hotspot_mask(
        mask,
        immutable_mask,
        von_mises,
        quantile=hotspot_quantile,
        dilate=hotspot_dilate,
    )
    union_mask = boundary_mask | hotspot_mask
    return boundary_mask, hotspot_mask, union_mask


def compute_direct_action_mask(
    catalog: list[DirectGridAction],
    union_mask: np.ndarray,
) -> np.ndarray:
    valid = np.zeros(len(catalog), dtype=bool)
    for idx, action in enumerate(catalog):
        if action.kind == "stop":
            valid[idx] = True
            continue
        valid[idx] = bool(union_mask[action.row, action.col])
    return valid


def apply_direct_action(
    mask: np.ndarray,
    action: DirectGridAction,
    immutable_mask: np.ndarray,
) -> np.ndarray:
    binary = ensure_binary(mask).copy()
    if action.kind == "stop":
        return binary
    if immutable_mask[action.row, action.col] == 0:
        binary[action.row, action.col] = 0
    return np.maximum(binary, immutable_mask).astype(np.uint8)


def _normalize_field(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    max_value = float(np.max(field)) if field.size else 0.0
    if max_value <= 1e-12:
        return np.zeros_like(field, dtype=np.float32)
    return field / max_value


def stack_direct_observation(
    mask: np.ndarray,
    immutable_mask: np.ndarray,
    boundary_mask: np.ndarray,
    hotspot_mask: np.ndarray,
    von_mises: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            ensure_binary(mask).astype(np.float32),
            np.asarray(immutable_mask, dtype=np.float32),
            np.asarray(boundary_mask, dtype=np.float32),
            np.asarray(hotspot_mask, dtype=np.float32),
            _normalize_field(von_mises),
        ],
        axis=0,
    )


if th is not None:  # pragma: no cover - optional dependency
    class SmallBinaryMaskCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: Any, features_dim: int = 128):
            super().__init__(observation_space, features_dim)
            channels = observation_space.shape[0]
            self.network = th.nn.Sequential(
                th.nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
                th.nn.ReLU(),
                th.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                th.nn.ReLU(),
                th.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                th.nn.ReLU(),
                th.nn.Flatten(),
#                 th.nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
#                 th.nn.ReLU(),
#                 th.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#                 th.nn.ReLU(),
#                 th.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#                 th.nn.ReLU(),
#                 th.nn.Flatten(),
            )
            with th.no_grad():
                sample = th.as_tensor(observation_space.sample()[None]).float()
                latent_dim = self.network(sample).shape[1]
            self.projection = th.nn.Sequential(th.nn.Linear(latent_dim, features_dim), th.nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.projection(self.network(observations.float()))


def default_policy_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"net_arch": [128, 128]}
#     kwargs: dict[str, Any] = {"net_arch": [256, 256]}
    if th is not None:
        kwargs["features_extractor_class"] = SmallBinaryMaskCNN
        kwargs["features_extractor_kwargs"] = {"features_dim": 128}
        # kwargs["features_extractor_kwargs"] = {"features_dim": 256}
    return kwargs


if gym is not None:  # pragma: no cover - optional dependency
    class BinaryTopologyRefineEnv(gym.Env):
        metadata = {"render_modes": ["human"]}

        def __init__(self, seed_mask: np.ndarray, config: ProblemConfig, evaluator: Evaluator):
            super().__init__()
            self.config = config
            self.evaluator = evaluator
            self.seed_mask = ensure_binary(seed_mask)
            self.mask = self.seed_mask.copy()
            self.catalog = build_action_catalog(self.mask.shape[0])
            support_load = evaluator.setups[self.mask.shape[0]].support_mask + evaluator.setups[self.mask.shape[0]].load_mask
            self.support_load_mask = np.clip(support_load, 0, 1).astype(np.float32)
            self.immutable_mask = self.support_load_mask.astype(np.uint8)
            self.full_eval_calls = 0
            self.stage_full_eval_calls = 0
            self.step_count = 0
            self.last_full_score = np.inf
            self.action_space = spaces.Discrete(len(self.catalog))
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, self.mask.shape[0], self.mask.shape[1]),
                dtype=np.float32,
            )

        def _observation(self) -> np.ndarray:
            return stack_observation(self.mask, self.support_load_mask, frontier_width=self.config.frontier_width).astype(np.float32)

        def action_masks(self) -> np.ndarray:
            return compute_action_mask(self.mask, self.catalog, self.immutable_mask, self.config.frontier_width)

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            self.mask = self.seed_mask.copy()
            self.step_count = 0
            self.full_eval_calls = 0
            self.last_full_score = np.inf
            return self._observation(), {}

        def step(self, action: int):
            masks = self.action_masks()
            if not masks[action]:
                reward = -1.0
                self.step_count += 1
                terminated = self.step_count >= self.config.max_episode_steps
                return self._observation(), reward, terminated, False, {"invalid_action": True}

            self.mask = apply_action(self.mask, self.catalog[action])
            self.step_count += 1

            intermediate = self._intermediate_reward()
            terminated = self.step_count >= self.config.max_episode_steps
            info: dict[str, Any] = {}
            reward = intermediate
            should_eval = self.step_count % self.config.full_eval_every == 0 or terminated
            can_eval = self.stage_full_eval_calls < self.config.max_rl_full_evals
            if should_eval and can_eval:
                evaluation = self.evaluator.evaluate(self.mask, "full64")
                self.full_eval_calls += 1
                self.stage_full_eval_calls += 1
                self.last_full_score = evaluation.score
                reward = self._terminal_reward(evaluation)
                info["evaluation"] = evaluation
            return self._observation(), float(reward), terminated, False, info

        def _intermediate_reward(self) -> float:
            volume_gap = abs(volume_fraction(self.mask) - self.config.volume_target)
            smoothness = calculate_smoothness_metric(self.mask) / max(self.mask.size, 1)
            islands = count_islands(self.mask) - 1
            return float(-(volume_gap + 0.05 * smoothness + 0.10 * max(islands, 0)))

        def _terminal_reward(self, evaluation) -> float:
            if not evaluation.passed_filters:
                return -2.0
            return float(1.0 / (1.0 + evaluation.score))

        def render(self):
            return self.mask.copy()


    class DirectBinaryTopologyRefineEnv(gym.Env):
        metadata = {"render_modes": ["human"]}

        def __init__(self, seed_mask: np.ndarray, config: ProblemConfig, evaluator: Evaluator):
            super().__init__()
            self.config = config
            self.evaluator = evaluator
            self.seed_mask = ensure_binary(seed_mask)
            self.mask = self.seed_mask.copy()
            self.catalog = build_direct_action_catalog(self.mask.shape[0])
            support_load = evaluator.setups[self.mask.shape[0]].support_mask + evaluator.setups[self.mask.shape[0]].load_mask
            self.support_load_mask = np.clip(support_load, 0, 1).astype(np.uint8)
            self.immutable_mask = self.support_load_mask.astype(np.uint8)
            self.action_space = spaces.Discrete(len(self.catalog))
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5, self.mask.shape[0], self.mask.shape[1]),
                dtype=np.float32,
            )
            self.step_count = 0
            self.full_eval_calls = 0
            self.stage_full_eval_calls = 0
            self.last_evaluation, self.last_fields = evaluator.evaluate_with_fields(self.mask, "full64")
            self.boundary_mask = np.zeros_like(self.mask, dtype=bool)
            self.hotspot_mask = np.zeros_like(self.mask, dtype=bool)
            self.union_mask = np.zeros_like(self.mask, dtype=bool)
            self.initial_candidate_counts = {"boundary": 0, "hotspot": 0, "union": 0}
            self.accepted_removals = 0
            self.rejected_invalid_removals = 0
            self.rejected_non_improving_removals = 0
            self.rejected_heuristic_regressions = 0
            self.accepted_source_breakdown = {"boundary": 0, "hotspot": 0, "both": 0}
            self.stop_used = False
            self.last_action_source = "none"
            self._refresh_editable_masks(self.mask, self.last_fields)

        def _observation(self) -> np.ndarray:
            return stack_direct_observation(
                self.mask,
                self.immutable_mask,
                self.boundary_mask,
                self.hotspot_mask,
                self.last_fields.von_mises,
            ).astype(np.float32)

        def action_masks(self) -> np.ndarray:
            return compute_direct_action_mask(self.catalog, self.union_mask)

        def _refresh_editable_masks(self, mask: np.ndarray, fields: ElementFieldDiagnostics) -> None:
            self.boundary_mask, self.hotspot_mask, self.union_mask = compute_direct_editable_masks(
                mask,
                self.immutable_mask,
                fields.von_mises,
                boundary_depth=self.config.rl_boundary_depth,
                hotspot_quantile=self.config.rl_stress_hotspot_quantile,
                hotspot_dilate=self.config.rl_stress_hotspot_dilate,
            )

        def _source_for_action(self, action: DirectGridAction) -> str:
            if action.kind == "stop":
                return "stop"
            on_boundary = bool(self.boundary_mask[action.row, action.col])
            on_hotspot = bool(self.hotspot_mask[action.row, action.col])
            if on_boundary and on_hotspot:
                return "both"
            if on_boundary:
                return "boundary"
            if on_hotspot:
                return "hotspot"
            return "none"

        def _diagnostics(self) -> dict[str, Any]:
            return {
                "boundary_candidate_count": int(self.boundary_mask.sum()),
                "hotspot_candidate_count": int(self.hotspot_mask.sum()),
                "union_candidate_count": int(self.union_mask.sum()),
                "initial_boundary_candidate_count": int(self.initial_candidate_counts["boundary"]),
                "initial_hotspot_candidate_count": int(self.initial_candidate_counts["hotspot"]),
                "initial_union_candidate_count": int(self.initial_candidate_counts["union"]),
                "accepted_removals": int(self.accepted_removals),
                "rejected_invalid_removals": int(self.rejected_invalid_removals),
                "rejected_non_improving_removals": int(self.rejected_non_improving_removals),
                "rejected_heuristic_regressions": int(self.rejected_heuristic_regressions),
                "accepted_source_breakdown": dict(self.accepted_source_breakdown),
                "last_action_source": self.last_action_source,
                "stop_used": bool(self.stop_used),
                "exact_full_evals_used": int(self.stage_full_eval_calls),
            }

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            self.mask = self.seed_mask.copy()
            self.step_count = 0
            self.full_eval_calls = 0
            self.stage_full_eval_calls = 0
            self.accepted_removals = 0
            self.rejected_invalid_removals = 0
            self.rejected_non_improving_removals = 0
            self.rejected_heuristic_regressions = 0
            self.accepted_source_breakdown = {"boundary": 0, "hotspot": 0, "both": 0}
            self.stop_used = False
            self.last_action_source = "none"
            self.last_evaluation, self.last_fields = self.evaluator.evaluate_with_fields(self.mask, "full64")
            self._refresh_editable_masks(self.mask, self.last_fields)
            self.initial_candidate_counts = {
                "boundary": int(self.boundary_mask.sum()),
                "hotspot": int(self.hotspot_mask.sum()),
                "union": int(self.union_mask.sum()),
            }
            return self._observation(), {"evaluation": self.last_evaluation, "rl_diagnostics": self._diagnostics()}

        def _heuristic_reward(self, mask: np.ndarray | None = None) -> float:
            binary = self.mask if mask is None else ensure_binary(mask)
            volume_gap = abs(volume_fraction(binary) - self.config.volume_target)
            smoothness = calculate_smoothness_metric(binary) / max(binary.size, 1)
            islands = max(count_islands(binary) - 1, 0)
            return float(-(volume_gap + 0.05 * smoothness + 0.10 * islands))

        def step(self, action: int):
            masks = self.action_masks()
            self.step_count += 1
            if not masks[action]:
                terminated = self.step_count >= min(self.config.max_episode_steps, 64)
                return self._observation(), -1.0, terminated, False, {"invalid_action": True}

            selected_action = self.catalog[action]
            self.last_action_source = self._source_for_action(selected_action)
            if selected_action.kind == "stop":
                self.stop_used = True
                editable_candidates = int(self.union_mask.sum())
                stop_penalty_applied = editable_candidates > 0 and self.accepted_removals == 0 and self.config.rl_stop_penalty > 0.0
                reward = -float(self.config.rl_stop_penalty) if stop_penalty_applied else 0.0
                info = {
                    "evaluation": self.last_evaluation,
                    "stopped": True,
                    "stop_penalty_applied": bool(stop_penalty_applied),
                    "editable_candidates_at_stop": editable_candidates,
                    "rl_diagnostics": self._diagnostics(),
                }
                return self._observation(), reward, True, False, info

            previous_mask = self.mask.copy()
            previous_evaluation = self.last_evaluation
            previous_fields = self.last_fields
            previous_heuristic = self._heuristic_reward(previous_mask)
            candidate_mask = apply_direct_action(
                previous_mask,
                selected_action,
                immutable_mask=self.immutable_mask,
            )
            info: dict[str, Any] = {}
            if self.stage_full_eval_calls < self.config.max_rl_full_evals:
                evaluation, fields = self.evaluator.evaluate_with_fields(candidate_mask, "full64")
                self.full_eval_calls += 1
                self.stage_full_eval_calls += 1
                if not evaluation.passed_filters:
                    self.mask = previous_mask
                    self.last_fields = previous_fields
                    reward = -1.0
                    info["reverted"] = True
                    info["revert_reason"] = "invalid_candidate"
                    self.rejected_invalid_removals += 1
                elif not previous_evaluation.passed_filters:
                    self.mask = candidate_mask
                    reward = float(1.0 / (1.0 + evaluation.score))
                    self.last_evaluation = evaluation
                    self.last_fields = fields
                    self._refresh_editable_masks(self.mask, self.last_fields)
                    self.accepted_removals += 1
                    if self.last_action_source in self.accepted_source_breakdown:
                        self.accepted_source_breakdown[self.last_action_source] += 1
                elif evaluation.score < previous_evaluation.score:
                    self.mask = candidate_mask
                    reward = float(
                        (previous_evaluation.score - evaluation.score) / max(abs(previous_evaluation.score), 1.0)
                    )
                    self.last_evaluation = evaluation
                    self.last_fields = fields
                    self._refresh_editable_masks(self.mask, self.last_fields)
                    self.accepted_removals += 1
                    if self.last_action_source in self.accepted_source_breakdown:
                        self.accepted_source_breakdown[self.last_action_source] += 1
                else:
                    self.mask = previous_mask
                    self.last_fields = previous_fields
                    reward = -float(
                        min(
                            (evaluation.score - previous_evaluation.score) / max(abs(previous_evaluation.score), 1.0),
                            1.0,
                        )
                    )
                    info["reverted"] = True
                    info["revert_reason"] = "non_improving_candidate"
                    self.rejected_non_improving_removals += 1
                info["evaluation"] = evaluation
            else:
                candidate_heuristic = self._heuristic_reward(candidate_mask)
                reward = float(candidate_heuristic - previous_heuristic)
                if candidate_heuristic >= previous_heuristic:
                    self.mask = candidate_mask
                    self._refresh_editable_masks(self.mask, previous_fields)
                    self.accepted_removals += 1
                    if self.last_action_source in self.accepted_source_breakdown:
                        self.accepted_source_breakdown[self.last_action_source] += 1
                else:
                    self.mask = previous_mask
                    info["reverted"] = True
                    info["revert_reason"] = "heuristic_regression"
                    self.rejected_heuristic_regressions += 1
            terminated = self.step_count >= min(self.config.max_episode_steps, 64)
            info["rl_diagnostics"] = self._diagnostics()
            return self._observation(), reward, terminated, False, info

        def render(self):
            return self.mask.copy()

        def rl_diagnostics(self) -> dict[str, Any]:
            return self._diagnostics()


def make_refine_env(seed_mask: np.ndarray, config: ProblemConfig, evaluator: Evaluator | None = None):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is required to build the refinement environment.")
    if evaluator is None:
        evaluator = Evaluator(config)
    return BinaryTopologyRefineEnv(seed_mask=seed_mask, config=config, evaluator=evaluator)


def make_direct64_refine_env(seed_mask: np.ndarray, config: ProblemConfig, evaluator: Evaluator | None = None):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is required to build the refinement environment.")
    if evaluator is None:
        evaluator = Evaluator(config)
    return DirectBinaryTopologyRefineEnv(seed_mask=seed_mask, config=config, evaluator=evaluator)
