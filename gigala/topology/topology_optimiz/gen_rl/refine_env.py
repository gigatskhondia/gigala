from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fem import Evaluator, ProblemConfig
from .metrics import calculate_smoothness_metric, count_islands, ensure_binary, prune_isolated_cells, volume_fraction
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


if th is not None:  # pragma: no cover - optional dependency
    class SmallBinaryMaskCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: Any, features_dim: int = 128):
            super().__init__(observation_space, features_dim)
            channels = observation_space.shape[0]
            self.network = th.nn.Sequential(
#                 th.nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
#                 th.nn.ReLU(),
#                 th.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#                 th.nn.ReLU(),
#                 th.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#                 th.nn.ReLU(),
#                 th.nn.Flatten(),
                th.nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
                th.nn.ReLU(),
                th.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                th.nn.ReLU(),
                th.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                th.nn.ReLU(),
                th.nn.Flatten(),
            )
            with th.no_grad():
                sample = th.as_tensor(observation_space.sample()[None]).float()
                latent_dim = self.network(sample).shape[1]
            self.projection = th.nn.Sequential(th.nn.Linear(latent_dim, features_dim), th.nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.projection(self.network(observations.float()))


def default_policy_kwargs() -> dict[str, Any]:
#     kwargs: dict[str, Any] = {"net_arch": [128, 128]}
    kwargs: dict[str, Any] = {"net_arch": [256, 256]}    
    if th is not None:
        kwargs["features_extractor_class"] = SmallBinaryMaskCNN
#         kwargs["features_extractor_kwargs"] = {"features_dim": 128}
        kwargs["features_extractor_kwargs"] = {"features_dim": 256}
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
            if self.step_count % self.config.full_eval_every == 0 or terminated:
                evaluation = self.evaluator.evaluate(self.mask, "full64")
                self.full_eval_calls += 1
                self.last_full_score = evaluation.score
                reward = self._terminal_reward(evaluation)
                info["evaluation"] = evaluation
                if self.full_eval_calls >= self.config.max_rl_full_evals:
                    terminated = True
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


def make_refine_env(seed_mask: np.ndarray, config: ProblemConfig, evaluator: Evaluator | None = None):
    if gym is None or spaces is None:
        raise ImportError("gymnasium is required to build the refinement environment.")
    if evaluator is None:
        evaluator = Evaluator(config)
    return BinaryTopologyRefineEnv(seed_mask=seed_mask, config=config, evaluator=evaluator)
