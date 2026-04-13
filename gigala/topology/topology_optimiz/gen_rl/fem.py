from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

from .metrics import (
    calculate_smoothness_metric,
    composite_penalty,
    count_islands,
    ensure_binary,
    prune_isolated_cells,
    retain_components_touching_region,
    touches_region,
    volume_fraction,
)
from .representation import infer_stage_resolutions


Fidelity = Literal["proxy16", "proxy32", "full64"]


@dataclass(slots=True)
class ProblemConfig:
    resolution: int
    volume_target: float = 0.55
    load_case: str = "cantilever"
    solver_backend: str = "scipy"
    runtime_budget_hours: float = 3.0
    random_seed: int = 42
    coarse_population: int = 128
    coarse_generations: int = 250
    coarse_elite_count: int = 16
    stage32_top_k: int = 8
    stage64_top_k: int = 2
    frontier_width: int = 2
    local_search_steps32: int = 48
    local_search_steps64: int = 96
    full_eval_every: int = 16
    max_episode_steps: int = 256
    enable_rl: bool = True
    rl_total_timesteps: int = 100_000
    max_full_evals: int = 20_000
    max_rl_full_evals: int = 5_000
    density_floor: float = 1e-4
    volume_tolerance: float = 0.20
    smoothness_weight: float = 0.10
    island_weight: float = 0.35
    volume_weight: float = 1.00
    one_island_bonus: float = 0.05
    invalid_penalty: float = 1e12
    proxy_filter_quantile: float = 0.35
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def stage_resolutions(self) -> tuple[int, ...]:
        return infer_stage_resolutions(self.resolution)


@dataclass(slots=True)
class EvalResult:
    fidelity: Fidelity
    resolution: int
    compliance: float
    score: float
    volume_fraction: float
    smoothness: int
    islands: int
    fea_performed: bool
    cache_hit: bool
    passed_filters: bool
    invalid_reason: str | None = None


@dataclass(slots=True)
class MeshSetup:
    resolution: int
    forces: np.ndarray
    freedofs: np.ndarray
    fixdofs: np.ndarray
    reduced_indices: np.ndarray
    keep_mask: np.ndarray
    index_map: np.ndarray
    stiffness_template: np.ndarray
    support_mask: np.ndarray
    load_mask: np.ndarray


def _mbb_beam(resolution: int, density: float) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros((resolution + 1, resolution + 1, 2), dtype=float)
    normals[0, 0, 0] = 1
    normals[0, 0, 1] = 1
    normals[0, -1, 0] = 1
    normals[0, -1, 1] = 1
    forces = np.zeros((resolution + 1, resolution + 1, 2), dtype=float)
    forces[-1, -1, 1] = -1
    return normals, forces


def _stiffness_matrix(young: float = 1.0, poisson: float = 0.3) -> np.ndarray:
    k = np.array(
        [
            1 / 2 - poisson / 6,
            1 / 8 + poisson / 8,
            -1 / 4 - poisson / 12,
            -1 / 8 + 3 * poisson / 8,
            -1 / 4 + poisson / 12,
            -1 / 8 - poisson / 8,
            poisson / 6,
            1 / 8 - 3 * poisson / 8,
        ],
        dtype=float,
    )
    return young / (1 - poisson**2) * np.array(
        [
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ]
    )


def _inverse_permutation(indices: np.ndarray) -> np.ndarray:
    inverse = np.zeros(len(indices), dtype=np.int64)
    inverse[indices] = np.arange(len(indices), dtype=np.int64)
    return inverse


def _build_setup(resolution: int, density_floor: float) -> MeshSetup:
    normals, forces = _mbb_beam(resolution, density_floor)
    fixdofs = np.flatnonzero(normals.ravel())
    alldofs = np.arange(2 * (resolution + 1) * (resolution + 1))
    freedofs = np.sort(np.setdiff1d(alldofs, fixdofs))

    ely, elx = np.meshgrid(np.arange(resolution), np.arange(resolution))
    ely = ely.reshape(-1, 1)
    elx = elx.reshape(-1, 1)

    n1 = (resolution + 1) * (elx + 0) + (ely + 0)
    n2 = (resolution + 1) * (elx + 1) + (ely + 0)
    n3 = (resolution + 1) * (elx + 1) + (ely + 1)
    n4 = (resolution + 1) * (elx + 0) + (ely + 1)
    edof = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1])
    edof = edof.T[0]
    x_list = np.repeat(edof, 8)
    y_list = np.tile(edof, 8).flatten()

    index_map = _inverse_permutation(np.concatenate([freedofs, fixdofs]))
    keep_mask = np.isin(x_list, freedofs) & np.isin(y_list, freedofs)
    reduced_i = index_map[y_list][keep_mask]
    reduced_j = index_map[x_list][keep_mask]
    reduced_indices = np.stack([reduced_i, reduced_j])

    support_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    support_mask[0, 0] = 1
    support_mask[0, -1] = 1
    load_mask = np.zeros((resolution, resolution), dtype=np.uint8)
    load_mask[-1, -1] = 1

    return MeshSetup(
        resolution=resolution,
        forces=forces.ravel(),
        freedofs=freedofs,
        fixdofs=fixdofs,
        reduced_indices=reduced_indices,
        keep_mask=keep_mask,
        index_map=index_map,
        stiffness_template=_stiffness_matrix(),
        support_mask=support_mask,
        load_mask=load_mask,
    )


class Evaluator:
    def __init__(self, config: ProblemConfig):
        self.config = config
        self.setups = {stage: _build_setup(stage, config.density_floor) for stage in config.stage_resolutions}
        self.cache: dict[tuple[Fidelity, str], EvalResult] = {}
        self.fea_counts = {"proxy16": 0, "proxy32": 0, "full64": 0}
        self.cache_hits = 0

    def _mask_digest(self, mask: np.ndarray) -> str:
        return hashlib.blake2b(np.ascontiguousarray(mask).view(np.uint8), digest_size=16).hexdigest()

    def _fidelity_resolution(self, fidelity: Fidelity) -> int:
        if fidelity == "proxy16":
            return 16
        if fidelity == "proxy32":
            return 32 if self.config.resolution >= 32 else self.config.resolution
        return self.config.resolution

    def _screen_geometry(self, mask: np.ndarray, setup: MeshSetup) -> tuple[bool, str | None]:
        volume = volume_fraction(mask)
        if volume == 0:
            return False, "empty_mask"
        if abs(volume - self.config.volume_target) > self.config.volume_tolerance:
            return False, "volume_out_of_range"
        if not touches_region(mask, setup.support_mask):
            return False, "missing_support_contact"
        if not touches_region(mask, setup.load_mask):
            return False, "missing_load_contact"
        if count_islands(mask) > max(6, setup.resolution // 4):
            return False, "too_many_islands"
        return True, None

    def _physical_density(self, mask: np.ndarray) -> np.ndarray:
        density = np.where(mask > 0, 1.0, self.config.density_floor)
        return scipy.ndimage.gaussian_filter(density, 1, mode="reflect")

    def _compliance(self, density: np.ndarray, displacements: np.ndarray, setup: MeshSetup) -> float:
        n = setup.resolution
        ely, elx = np.meshgrid(np.arange(n), np.arange(n))
        n1 = (n + 1) * (elx + 0) + (ely + 0)
        n2 = (n + 1) * (elx + 1) + (ely + 0)
        n3 = (n + 1) * (elx + 1) + (ely + 1)
        n4 = (n + 1) * (elx + 0) + (ely + 1)
        all_ixs = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1])
        u_selected = displacements[all_ixs]
        ke_u = np.einsum("ij,jkl->ikl", setup.stiffness_template, u_selected)
        ce = np.einsum("ijk,ijk->jk", u_selected, ke_u)
        return float(np.sum(density * ce.T))

    def _solve_displacements(self, density: np.ndarray, setup: MeshSetup) -> np.ndarray:
        kd = density.T.reshape(setup.resolution * setup.resolution, 1, 1)
        values = (kd * np.tile(setup.stiffness_template, kd.shape)).flatten()
        matrix = scipy.sparse.coo_matrix(
            (values[setup.keep_mask], setup.reduced_indices),
            shape=(setup.freedofs.size, setup.freedofs.size),
        ).tocsc()

        if self.config.solver_backend == "cholmod":
            try:
                from sksparse.cholmod import cholesky  # type: ignore

                solver = cholesky(matrix)
                reduced = solver(setup.forces[setup.freedofs])
            except Exception:
                reduced = scipy.sparse.linalg.splu(matrix).solve(setup.forces[setup.freedofs])
        else:
            reduced = scipy.sparse.linalg.splu(matrix).solve(setup.forces[setup.freedofs])
        full = np.concatenate([reduced, np.zeros(len(setup.fixdofs), dtype=float)])
        return full[setup.index_map]

    def _score(self, mask: np.ndarray, compliance: float) -> float:
        penalty = composite_penalty(
            mask,
            volume_target=self.config.volume_target,
            smoothness_weight=self.config.smoothness_weight,
            island_weight=self.config.island_weight,
            volume_weight=self.config.volume_weight,
        )
        if count_islands(mask) == 1:
            penalty = max(penalty - self.config.one_island_bonus, 0.0)
        return float(compliance * (1.0 + penalty))

    def evaluate(self, mask: np.ndarray, fidelity: Fidelity) -> EvalResult:
        resolution = self._fidelity_resolution(fidelity)
        if resolution not in self.setups:
            self.setups[resolution] = _build_setup(resolution, self.config.density_floor)
        setup = self.setups[resolution]

        binary = ensure_binary(mask)
        if binary.shape != (resolution, resolution):
            raise ValueError(f"Mask shape {binary.shape} does not match fidelity resolution {resolution}.")
        processed = prune_isolated_cells(binary)
        processed = np.maximum(processed, setup.support_mask)
        processed = np.maximum(processed, setup.load_mask)
        processed = retain_components_touching_region(processed, setup.support_mask)
        cache_key = (fidelity, self._mask_digest(processed))
        if cache_key in self.cache:
            self.cache_hits += 1
            cached = self.cache[cache_key]
            return dataclasses.replace(cached, cache_hit=True)

        passed, invalid_reason = self._screen_geometry(processed, setup)
        if not passed:
            result = EvalResult(
                fidelity=fidelity,
                resolution=resolution,
                compliance=self.config.invalid_penalty,
                score=self.config.invalid_penalty,
                volume_fraction=volume_fraction(processed),
                smoothness=calculate_smoothness_metric(processed),
                islands=count_islands(processed),
                fea_performed=False,
                cache_hit=False,
                passed_filters=False,
                invalid_reason=invalid_reason,
            )
            self.cache[cache_key] = result
            return result

        if fidelity == "full64" and self.fea_counts["full64"] >= self.config.max_full_evals:
            result = EvalResult(
                fidelity=fidelity,
                resolution=resolution,
                compliance=self.config.invalid_penalty,
                score=self.config.invalid_penalty,
                volume_fraction=volume_fraction(processed),
                smoothness=calculate_smoothness_metric(processed),
                islands=count_islands(processed),
                fea_performed=False,
                cache_hit=False,
                passed_filters=False,
                invalid_reason="full_eval_budget_exhausted",
            )
            self.cache[cache_key] = result
            return result

        try:
            density = self._physical_density(processed)
            displacements = self._solve_displacements(density, setup)
            compliance = self._compliance(density, displacements, setup)
            score = self._score(processed, compliance)
            self.fea_counts[fidelity] += 1
            result = EvalResult(
                fidelity=fidelity,
                resolution=resolution,
                compliance=compliance,
                score=score,
                volume_fraction=volume_fraction(processed),
                smoothness=calculate_smoothness_metric(processed),
                islands=count_islands(processed),
                fea_performed=True,
                cache_hit=False,
                passed_filters=True,
            )
        except Exception as exc:  # pragma: no cover - solver failure is data dependent
            result = EvalResult(
                fidelity=fidelity,
                resolution=resolution,
                compliance=self.config.invalid_penalty,
                score=self.config.invalid_penalty,
                volume_fraction=volume_fraction(processed),
                smoothness=calculate_smoothness_metric(processed),
                islands=count_islands(processed),
                fea_performed=False,
                cache_hit=False,
                passed_filters=False,
                invalid_reason=str(exc),
            )
        self.cache[cache_key] = result
        return result


def evaluation_snapshot(evaluator: Evaluator) -> dict[str, float]:
    return {
        "proxy16": float(evaluator.fea_counts["proxy16"]),
        "proxy32": float(evaluator.fea_counts["proxy32"]),
        "full64": float(evaluator.fea_counts["full64"]),
        "cache_hits": float(evaluator.cache_hits),
        "cache_size": float(len(evaluator.cache)),
    }
