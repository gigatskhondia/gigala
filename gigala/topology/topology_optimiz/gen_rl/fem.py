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


PipelineMode = Literal["multistage", "direct64_exact", "rl_only_exact"]
Fidelity = Literal["proxy16", "proxy32", "full64"]


@dataclass
class ProblemConfig:
    resolution: int
    pipeline_mode: PipelineMode = "multistage"
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
    direct_population: int = 48
    direct_elite_count: int = 8
    direct_offspring_batch: int = 16
    direct_archive_size: int = 32
    direct_restart_stagnation_evals: int = 400
    workers: int | str = "auto"
    full_eval_every: int = 16
    max_episode_steps: int = 256
    enable_rl: bool | None = None
    rl_device: str = "auto"
    rl_total_timesteps: int = 100_000
    rl_archive_top_k: int = 4
    rl_boundary_depth: int = 1
    rl_stress_metric: Literal["von_mises"] = "von_mises"
    rl_stress_hotspot_quantile: float = 0.95
    rl_stress_hotspot_dilate: int = 1
    rl_stop_penalty: float = 0.05
    rl_degenerate_episode_window: int = 32
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

    def __post_init__(self) -> None:
        if self.enable_rl is None:
            self.enable_rl = self.pipeline_mode in ("multistage", "rl_only_exact")
        if isinstance(self.workers, int) and self.workers < 1:
            self.workers = 1
        if self.max_full_evals < 0:
            raise ValueError("max_full_evals must be >= 0.")
        if self.max_rl_full_evals < 0:
            raise ValueError("max_rl_full_evals must be >= 0.")
        if self.rl_archive_top_k < 1:
            raise ValueError("rl_archive_top_k must be >= 1.")
        if self.rl_boundary_depth < 1:
            raise ValueError("rl_boundary_depth must be >= 1.")
        if self.rl_stress_metric != "von_mises":
            raise ValueError("rl_stress_metric must be 'von_mises'.")
        if not 0.0 < self.rl_stress_hotspot_quantile < 1.0:
            raise ValueError("rl_stress_hotspot_quantile must be between 0 and 1.")
        if self.rl_stress_hotspot_dilate < 0:
            raise ValueError("rl_stress_hotspot_dilate must be >= 0.")
        if self.rl_stop_penalty < 0.0:
            raise ValueError("rl_stop_penalty must be >= 0.")
        if self.rl_degenerate_episode_window < 0:
            raise ValueError("rl_degenerate_episode_window must be >= 0.")

    @property
    def stage_resolutions(self) -> tuple[int, ...]:
        if self.pipeline_mode in ("direct64_exact", "rl_only_exact"):
            return (self.resolution,)
        return infer_stage_resolutions(self.resolution)

    def has_runtime_budget(self) -> bool:
        return self.runtime_budget_hours > 0


@dataclass
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


@dataclass
class ElementFieldDiagnostics:
    von_mises: np.ndarray
    strain_energy_density: np.ndarray | None = None


@dataclass
class MeshSetup:
    resolution: int
    forces: np.ndarray
    freedofs: np.ndarray
    fixdofs: np.ndarray
    reduced_indices: np.ndarray
    keep_mask: np.ndarray
    index_map: np.ndarray
    element_dofs: np.ndarray
    stiffness_template: np.ndarray
    constitutive_matrix: np.ndarray
    strain_displacement_matrix: np.ndarray
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


def _constitutive_matrix(young: float = 1.0, poisson: float = 0.3) -> np.ndarray:
    return young / (1 - poisson**2) * np.array(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, (1 - poisson) / 2],
        ],
        dtype=float,
    )


def _strain_displacement_matrix() -> np.ndarray:
    dndx = np.array([-0.5, 0.5, 0.5, -0.5], dtype=float)
    dndy = np.array([-0.5, -0.5, 0.5, 0.5], dtype=float)
    matrix = np.zeros((3, 8), dtype=float)
    matrix[0, 0::2] = dndx
    matrix[1, 1::2] = dndy
    matrix[2, 0::2] = dndy
    matrix[2, 1::2] = dndx
    return matrix


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
    element_dofs = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1])
    element_dofs = element_dofs.T[0]
    x_list = np.repeat(element_dofs, 8)
    y_list = np.tile(element_dofs, 8).flatten()

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
        element_dofs=element_dofs,
        stiffness_template=_stiffness_matrix(),
        constitutive_matrix=_constitutive_matrix(),
        strain_displacement_matrix=_strain_displacement_matrix(),
        support_mask=support_mask,
        load_mask=load_mask,
    )


class Evaluator:
    def __init__(self, config: ProblemConfig):
        self.config = config
        self.setups = {stage: _build_setup(stage, config.density_floor) for stage in config.stage_resolutions}
        self.cache: dict[tuple[Fidelity, str], EvalResult] = {}
        self.field_cache: dict[tuple[Fidelity, str], ElementFieldDiagnostics] = {}
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

    def canonicalize(self, mask: np.ndarray, fidelity: Fidelity) -> tuple[np.ndarray, MeshSetup]:
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
        return processed, setup

    def cache_key_for(self, mask: np.ndarray, fidelity: Fidelity) -> tuple[tuple[Fidelity, str], np.ndarray, MeshSetup]:
        processed, setup = self.canonicalize(mask, fidelity)
        return (fidelity, self._mask_digest(processed)), processed, setup

    def cached_result(self, cache_key: tuple[Fidelity, str]) -> EvalResult | None:
        cached = self.cache.get(cache_key)
        if cached is None:
            return None
        self.cache_hits += 1
        return dataclasses.replace(cached, cache_hit=True)

    def _copy_field_diagnostics(self, fields: ElementFieldDiagnostics) -> ElementFieldDiagnostics:
        return ElementFieldDiagnostics(
            von_mises=np.array(fields.von_mises, copy=True),
            strain_energy_density=None
            if fields.strain_energy_density is None
            else np.array(fields.strain_energy_density, copy=True),
        )

    def _empty_field_diagnostics(self, resolution: int) -> ElementFieldDiagnostics:
        zeros = np.zeros((resolution, resolution), dtype=float)
        return ElementFieldDiagnostics(von_mises=zeros, strain_energy_density=zeros.copy())

    def screen_processed(self, processed: np.ndarray, fidelity: Fidelity, setup: MeshSetup | None = None) -> EvalResult | None:
        resolution = self._fidelity_resolution(fidelity)
        setup = setup or self.setups[resolution]
        passed, invalid_reason = self._screen_geometry(processed, setup)
        if not passed:
            return self.make_invalid_result(processed, fidelity, invalid_reason or "failed_filters")
        if fidelity == "full64" and self.fea_counts["full64"] >= self.config.max_full_evals:
            return self.make_invalid_result(processed, fidelity, "full_eval_budget_exhausted")
        return None

    def make_invalid_result(self, processed: np.ndarray, fidelity: Fidelity, reason: str) -> EvalResult:
        resolution = self._fidelity_resolution(fidelity)
        return EvalResult(
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
            invalid_reason=reason,
        )

    def store_result(self, processed: np.ndarray, fidelity: Fidelity, result: EvalResult) -> None:
        cache_key = (fidelity, self._mask_digest(processed))
        self.cache[cache_key] = dataclasses.replace(result, cache_hit=False)

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
        _, strain_energy_density = self._element_quantities(density, displacements, setup)
        return float(np.sum(strain_energy_density))

    def _element_quantities(
        self,
        density: np.ndarray,
        displacements: np.ndarray,
        setup: MeshSetup,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_selected = displacements[setup.element_dofs]
        ke_u = np.einsum("ij,ej->ei", setup.stiffness_template, u_selected)
        ce = np.einsum("ei,ei->e", u_selected, ke_u)
        ce_grid = ce.reshape(setup.resolution, setup.resolution).T
        return u_selected, density * ce_grid

    def _field_diagnostics(
        self,
        density: np.ndarray,
        u_selected: np.ndarray,
        strain_energy_density: np.ndarray,
        setup: MeshSetup,
    ) -> ElementFieldDiagnostics:
        strain = np.einsum("ij,ej->ei", setup.strain_displacement_matrix, u_selected)
        stress = np.einsum("ij,ej->ei", setup.constitutive_matrix, strain)
        stress *= density.T.reshape(-1, 1)
        sigma_x = stress[:, 0]
        sigma_y = stress[:, 1]
        tau_xy = stress[:, 2]
        von_mises = np.sqrt(np.maximum(sigma_x**2 - sigma_x * sigma_y + sigma_y**2 + 3.0 * tau_xy**2, 0.0))
        return ElementFieldDiagnostics(
            von_mises=von_mises.reshape(setup.resolution, setup.resolution).T,
            strain_energy_density=np.asarray(strain_energy_density, dtype=float),
        )

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
        cache_key, processed, setup = self.cache_key_for(mask, fidelity)
        cached = self.cached_result(cache_key)
        if cached is not None:
            return cached

        invalid = self.screen_processed(processed, fidelity, setup=setup)
        if invalid is not None:
            self.cache[cache_key] = invalid
            return invalid

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

    def evaluate_with_fields(self, mask: np.ndarray, fidelity: Fidelity) -> tuple[EvalResult, ElementFieldDiagnostics]:
        resolution = self._fidelity_resolution(fidelity)
        cache_key, processed, setup = self.cache_key_for(mask, fidelity)
        cached_result = self.cache.get(cache_key)
        cached_fields = self.field_cache.get(cache_key)
        if cached_result is not None and (cached_fields is not None or not cached_result.passed_filters):
            self.cache_hits += 1
            result = dataclasses.replace(cached_result, cache_hit=True)
            fields = self._copy_field_diagnostics(cached_fields) if cached_fields is not None else self._empty_field_diagnostics(resolution)
            return result, fields

        invalid = self.screen_processed(processed, fidelity, setup=setup)
        if invalid is not None:
            self.cache[cache_key] = invalid
            return invalid, self._empty_field_diagnostics(resolution)

        try:
            density = self._physical_density(processed)
            displacements = self._solve_displacements(density, setup)
            u_selected, strain_energy_density = self._element_quantities(density, displacements, setup)
            compliance = float(np.sum(strain_energy_density))
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
            fields = self._field_diagnostics(density, u_selected, strain_energy_density, setup)
            self.cache[cache_key] = result
            self.field_cache[cache_key] = self._copy_field_diagnostics(fields)
            return result, fields
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
            return result, self._empty_field_diagnostics(resolution)


def evaluation_snapshot(evaluator: Evaluator) -> dict[str, float]:
    return {
        "proxy16": float(evaluator.fea_counts["proxy16"]),
        "proxy32": float(evaluator.fea_counts["proxy32"]),
        "full64": float(evaluator.fea_counts["full64"]),
        "cache_hits": float(evaluator.cache_hits),
        "cache_size": float(len(evaluator.cache)),
    }
