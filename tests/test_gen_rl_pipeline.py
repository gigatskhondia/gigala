from __future__ import annotations

import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gigala.topology.topology_optimiz.gen_rl import ProblemConfig, run_direct64_exact_search, run_multistage_search
from gigala.topology.topology_optimiz.gen_rl.cli import _save_outputs, main as cli_main
from gigala.topology.topology_optimiz.gen_rl.direct_search import _init_worker, build_mutation_coverage, evaluate_exact_batch
from gigala.topology.topology_optimiz.gen_rl.fem import ElementFieldDiagnostics, EvalResult, Evaluator
from gigala.topology.topology_optimiz.gen_rl.pipeline import DirectRLDegeneracyMonitor, _resolve_rl_device
import gigala.topology.topology_optimiz.gen_rl.pipeline as pipeline_module
from gigala.topology.topology_optimiz.gen_rl.refine_env import (
    build_action_catalog,
    build_direct_action_catalog,
    compute_action_mask,
    compute_direct_action_mask,
    compute_direct_editable_masks,
    gym,
    make_direct64_refine_env,
)
from gigala.topology.topology_optimiz.gen_rl.representation import upsample_binary_mask


class RepresentationTests(unittest.TestCase):
    def test_upsample_preserves_binary_values(self) -> None:
        coarse = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
        up_8 = upsample_binary_mask(coarse, 8)
        up_16 = upsample_binary_mask(up_8, 16)
        self.assertEqual(up_8.shape, (8, 8))
        self.assertEqual(up_16.shape, (16, 16))
        self.assertEqual(set(np.unique(up_16).tolist()), {0, 1})


class EvaluatorTests(unittest.TestCase):
    def make_mask(self, resolution: int) -> np.ndarray:
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        mask[0, :] = 1
        mask[:, -1] = 1
        mask[: 3 * resolution // 4, resolution // 4 :] = 1
        mask[0, 0] = 1
        mask[0, -1] = 1
        mask[-1, -1] = 1
        return mask

    def test_cached_and_uncached_evaluation_match(self) -> None:
        config = ProblemConfig(resolution=64, enable_rl=False, max_full_evals=50)
        evaluator = Evaluator(config)
        mask = self.make_mask(64)

        first = evaluator.evaluate(mask, "full64")
        second = evaluator.evaluate(mask, "full64")

        self.assertTrue(first.passed_filters)
        self.assertTrue(second.cache_hit)
        self.assertAlmostEqual(first.compliance, second.compliance)
        self.assertAlmostEqual(first.score, second.score)

    def test_evaluate_with_fields_is_cached_by_same_digest(self) -> None:
        config = ProblemConfig(resolution=64, enable_rl=False, max_full_evals=50)
        evaluator = Evaluator(config)
        mask = self.make_mask(64)

        first_eval, first_fields = evaluator.evaluate_with_fields(mask, "full64")
        second_eval, second_fields = evaluator.evaluate_with_fields(mask, "full64")

        self.assertTrue(first_eval.passed_filters)
        self.assertTrue(second_eval.cache_hit)
        self.assertEqual(first_fields.von_mises.shape, (64, 64))
        self.assertTrue(np.allclose(first_fields.von_mises, second_fields.von_mises))

    def test_exact_compliance_matches_force_displacement_work(self) -> None:
        config = ProblemConfig(resolution=64, enable_rl=False, max_full_evals=50)
        evaluator = Evaluator(config)
        mask = self.make_mask(64)
        processed, setup = evaluator.canonicalize(mask, "full64")
        density = evaluator._physical_density(processed)
        displacements = evaluator._solve_displacements(density, setup)

        compliance = evaluator._compliance(density, displacements, setup)
        external_work = float(setup.forces @ displacements)

        self.assertGreater(compliance, 0.0)
        self.assertAlmostEqual(compliance, external_work, places=3)

    def test_zero_max_full_evals_blocks_exact_evaluation(self) -> None:
        config = ProblemConfig(resolution=64, enable_rl=False, max_full_evals=0)
        evaluator = Evaluator(config)
        mask = self.make_mask(64)

        first = evaluator.evaluate(mask, "full64")
        self.assertFalse(first.passed_filters)
        self.assertFalse(first.fea_performed)
        self.assertEqual(first.invalid_reason, "full_eval_budget_exhausted")
        self.assertEqual(evaluator.fea_counts["full64"], 0)

    def test_proxy_ranking_correlates_with_upsampled_full_ranking(self) -> None:
        config = ProblemConfig(resolution=32, enable_rl=False, max_full_evals=200)
        evaluator = Evaluator(config)
        rng = np.random.default_rng(0)
        proxy_scores: list[float] = []
        full_scores: list[float] = []

        for _ in range(6):
            mask16 = np.zeros((16, 16), dtype=np.uint8)
            mask16[0, :] = 1
            mask16[:, -1] = 1
            mask16[:12, 6:] = (rng.random((12, 10)) > 0.35).astype(np.uint8)
            mask16[0, 0] = 1
            mask16[0, -1] = 1
            mask16[-1, -1] = 1
            proxy_scores.append(evaluator.evaluate(mask16, "proxy16").score)
            full_scores.append(evaluator.evaluate(upsample_binary_mask(mask16, 32), "proxy32").score)

        proxy_order = np.argsort(proxy_scores)
        full_order = np.argsort(full_scores)
        agreement = np.mean(proxy_order == full_order)
        self.assertGreaterEqual(agreement, 0.5)

    def test_resolve_rl_device_prefers_mps_for_auto(self) -> None:
        class FakeMPS:
            @staticmethod
            def is_available() -> bool:
                return True

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class FakeBackends:
            mps = FakeMPS()

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

        config = ProblemConfig(resolution=64, rl_device="auto")
        self.assertEqual(_resolve_rl_device(config, torch_module=FakeTorch), "mps")


class PipelineTests(unittest.TestCase):
    def make_mask(self, resolution: int) -> np.ndarray:
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        mask[0, :] = 1
        mask[:, -1] = 1
        mask[resolution // 5 :, resolution // 2 :] = 1
        mask[0, 0] = 1
        mask[0, -1] = 1
        mask[-1, -1] = 1
        return mask

    def test_multistage_pipeline_returns_stage_artifacts(self) -> None:
        config = ProblemConfig(
            resolution=64,
            enable_rl=False,
            coarse_population=10,
            coarse_generations=3,
            coarse_elite_count=4,
            stage32_top_k=4,
            stage64_top_k=2,
            local_search_steps32=3,
            local_search_steps64=3,
            runtime_budget_hours=0.02,
            max_full_evals=80,
        )
        artifacts = run_multistage_search(config)
        self.assertEqual(artifacts.coarse16.shape, (16, 16))
        self.assertEqual(artifacts.refined32.shape, (32, 32))
        self.assertEqual(artifacts.refined64.shape, (64, 64))
        self.assertIn("coarse16", artifacts.metrics)
        self.assertIn("refined32", artifacts.metrics)
        self.assertIn("refined64", artifacts.metrics)
        self.assertGreater(artifacts.fea_counts["proxy16"], 0.0)
        self.assertGreater(artifacts.fea_counts["proxy32"], 0.0)
        self.assertGreater(artifacts.fea_counts["full64"], 0.0)

    def test_direct_pipeline_returns_exact_only_artifacts(self) -> None:
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=False,
            runtime_budget_hours=0.001,
            direct_population=4,
            direct_elite_count=2,
            direct_offspring_batch=2,
            direct_archive_size=3,
            direct_restart_stagnation_evals=12,
            max_full_evals=12,
            workers=1,
        )
        artifacts = run_direct64_exact_search(config)
        self.assertEqual(len(artifacts.initial_population), 4)
        self.assertEqual(artifacts.best64.shape, (64, 64))
        self.assertGreaterEqual(len(artifacts.archive_best), 1)
        self.assertIn("seed64", artifacts.metrics)
        self.assertIn("best64", artifacts.metrics)
        self.assertEqual(artifacts.fea_counts["proxy16"], 0.0)
        self.assertEqual(artifacts.fea_counts["proxy32"], 0.0)
        self.assertGreater(artifacts.fea_counts["full64"], 0.0)

    def test_runtime_budget_zero_uses_full_eval_budget_as_stop_factor(self) -> None:
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=False,
            runtime_budget_hours=0,
            direct_population=4,
            direct_elite_count=2,
            direct_offspring_batch=2,
            direct_archive_size=3,
            direct_restart_stagnation_evals=12,
            max_full_evals=12,
            workers=1,
        )
        artifacts = run_direct64_exact_search(config)
        self.assertEqual(artifacts.fea_counts["full64"], 12.0)
        self.assertIn("direct64 full64 evaluation budget exhausted", artifacts.warnings)

    def test_direct_mutation_coverage_reaches_every_cell(self) -> None:
        coverage = build_mutation_coverage(64)
        self.assertEqual(coverage.shape, (64, 64))
        self.assertTrue(bool(coverage.all()))

    def test_parallel_exact_batch_matches_serial(self) -> None:
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=False,
            max_full_evals=20,
            workers=2,
        )
        masks = [self.make_mask(64), np.rot90(self.make_mask(64))]
        serial_evaluator = Evaluator(config)
        serial = evaluate_exact_batch(masks, evaluator=serial_evaluator, config=config, workers=1, pool=None)
        parallel_evaluator = Evaluator(config)
        with ProcessPoolExecutor(max_workers=2, initializer=_init_worker, initargs=(config,)) as pool:
            parallel = evaluate_exact_batch(masks, evaluator=parallel_evaluator, config=config, workers=2, pool=pool)
        serial_scores = [candidate.evaluation.score for candidate in serial]
        parallel_scores = [candidate.evaluation.score for candidate in parallel]
        self.assertEqual(len(serial_scores), len(parallel_scores))
        self.assertEqual(serial_scores, parallel_scores)

    def test_frontier_action_mask_exposes_valid_actions(self) -> None:
        resolution = 32
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        mask[0, :] = 1
        mask[:, -1] = 1
        mask[8:20, 12:18] = 1
        immutable = np.zeros_like(mask, dtype=np.uint8)
        immutable[0, 0] = 1
        immutable[0, -1] = 1
        immutable[-1, -1] = 1
        catalog = build_action_catalog(resolution)
        action_mask = compute_action_mask(mask, catalog, immutable, frontier_width=2)
        self.assertEqual(action_mask.shape[0], len(catalog))
        self.assertGreater(int(action_mask.sum()), 0)

    def test_direct_action_mask_blocks_material_removal_below_volume_band(self) -> None:
        resolution = 8
        mask = np.ones((resolution, resolution), dtype=np.uint8)
        mask[3:5, 3:5] = 0
        immutable = np.zeros_like(mask, dtype=np.uint8)
        immutable[0, 0] = 1
        immutable[0, -1] = 1
        immutable[-1, -1] = 1
        stress = np.zeros((resolution, resolution), dtype=float)
        stress[2, 2] = 10.0
        boundary_mask, hotspot_mask, union_mask = compute_direct_editable_masks(
            mask,
            immutable,
            stress,
            boundary_depth=1,
            hotspot_quantile=0.95,
            hotspot_dilate=1,
        )
        catalog = build_direct_action_catalog(resolution)
        action_mask = compute_direct_action_mask(catalog, union_mask)
        self.assertEqual(action_mask.shape[0], len(catalog))
        self.assertTrue(boundary_mask[0, 3])
        self.assertTrue(boundary_mask[2, 3])  # internal hole contour
        self.assertFalse(boundary_mask[1, 1])
        self.assertTrue(hotspot_mask[2, 2])
        self.assertTrue(union_mask[2, 2])
        center_action = next(
            idx for idx, action in enumerate(catalog) if action.kind == "remove_cell" and action.row == 2 and action.col == 2
        )
        interior_action = next(
            idx for idx, action in enumerate(catalog) if action.kind == "remove_cell" and action.row == 1 and action.col == 1
        )
        stop_action = next(idx for idx, action in enumerate(catalog) if action.kind == "stop")
        self.assertTrue(action_mask[center_action])
        self.assertFalse(action_mask[interior_action])
        self.assertTrue(action_mask[stop_action])

    def test_direct_env_starts_from_native_seed(self) -> None:
        if gym is None:
            self.skipTest("gymnasium is unavailable")
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=False,
            max_full_evals=40,
            max_rl_full_evals=4,
        )
        seed = self.make_mask(64)
        env = make_direct64_refine_env(seed, config)
        observation, info = env.reset()
        self.assertEqual(observation.shape, (5, 64, 64))
        self.assertIn("evaluation", info)
        self.assertIn("rl_diagnostics", info)
        action_masks = env.action_masks()
        self.assertGreater(int(action_masks.sum()), 0)
        action = int(np.flatnonzero(action_masks[:-1])[0])
        next_observation, _reward, _terminated, _truncated, info = env.step(action)
        self.assertEqual(next_observation.shape, (5, 64, 64))
        self.assertIn("evaluation", info)
        self.assertIn("rl_diagnostics", info)

    def test_direct_env_reverts_non_improving_exact_candidate(self) -> None:
        if gym is None:
            self.skipTest("gymnasium is unavailable")
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=False,
            max_full_evals=40,
            max_rl_full_evals=4,
        )
        seed = self.make_mask(64)
        env = make_direct64_refine_env(seed, config)
        initial_mask = env.mask.copy()
        action_masks = env.action_masks()
        action = int(np.flatnonzero(action_masks[:-1])[0])
        worse_eval = EvalResult(
            fidelity="full64",
            resolution=64,
            compliance=1_000_000_000_000.0,
            score=1_000_000_000_000.0,
            volume_fraction=0.57,
            smoothness=1_500,
            islands=1,
            fea_performed=True,
            cache_hit=False,
            passed_filters=True,
        )
        zero_fields = ElementFieldDiagnostics(von_mises=np.zeros((64, 64), dtype=float), strain_energy_density=np.zeros((64, 64), dtype=float))
        with patch.object(env.evaluator, "evaluate_with_fields", return_value=(worse_eval, zero_fields)):
            _obs, reward, _terminated, _truncated, info = env.step(action)
        self.assertTrue(np.array_equal(env.mask, initial_mask))
        self.assertLess(reward, 0.0)
        self.assertTrue(info["reverted"])
        self.assertEqual(info["revert_reason"], "non_improving_candidate")

    def test_zero_max_rl_full_evals_skips_exact_rl_reward(self) -> None:
        if gym is None:
            self.skipTest("gymnasium is unavailable")
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=False,
            max_full_evals=40,
            max_rl_full_evals=0,
            max_episode_steps=64,
        )
        seed = self.make_mask(64)
        env = make_direct64_refine_env(seed, config)
        env.reset()
        action_masks = env.action_masks()
        action = int(np.flatnonzero(action_masks[:-1])[0])
        _obs, _reward, terminated, _truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertNotIn("evaluation", info)
        self.assertIn("rl_diagnostics", info)

    def test_direct_env_stop_action_terminates(self) -> None:
        if gym is None:
            self.skipTest("gymnasium is unavailable")
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=False,
            max_full_evals=40,
            max_rl_full_evals=4,
        )
        env = make_direct64_refine_env(self.make_mask(64), config)
        env.reset()
        stop_idx = len(env.catalog) - 1
        _obs, reward, terminated, _truncated, info = env.step(stop_idx)
        self.assertTrue(terminated)
        self.assertLess(reward, 0.0)
        self.assertTrue(info["stopped"])
        self.assertTrue(info["stop_penalty_applied"])

    def test_direct_rl_degeneracy_monitor_triggers_on_repeated_immediate_stops(self) -> None:
        monitor = DirectRLDegeneracyMonitor(episode_window=3)

        episode_info = {
            "episode": {"l": 1, "r": -0.05},
            "stop_penalty_applied": True,
            "rl_diagnostics": {
                "stop_used": True,
                "accepted_removals": 0,
            },
        }

        self.assertFalse(monitor.observe_episode(episode_info))
        self.assertFalse(monitor.observe_episode(episode_info))
        self.assertTrue(monitor.observe_episode(episode_info))
        snapshot = monitor.snapshot()
        self.assertTrue(snapshot["stopped_early"])
        self.assertEqual(snapshot["reason"], "degenerate_immediate_stop_policy:3_episodes")

    def test_direct_rl_degeneracy_monitor_ignores_non_degenerate_episode(self) -> None:
        monitor = DirectRLDegeneracyMonitor(episode_window=2)

        first = {
            "episode": {"l": 1, "r": -0.05},
            "stop_penalty_applied": True,
            "rl_diagnostics": {
                "stop_used": True,
                "accepted_removals": 0,
            },
        }
        second = {
            "episode": {"l": 4, "r": 0.02},
            "stop_penalty_applied": False,
            "rl_diagnostics": {
                "stop_used": False,
                "accepted_removals": 1,
            },
        }

        self.assertFalse(monitor.observe_episode(first))
        self.assertFalse(monitor.observe_episode(second))
        snapshot = monitor.snapshot()
        self.assertFalse(snapshot["stopped_early"])

    def test_monotonic_selector_accepts_only_better_valid_candidate(self) -> None:
        incumbent_mask = self.make_mask(64)
        candidate_mask = np.rot90(incumbent_mask)
        incumbent = EvalResult(
            fidelity="full64",
            resolution=64,
            compliance=30.0,
            score=30.0,
            volume_fraction=0.56,
            smoothness=100,
            islands=1,
            fea_performed=True,
            cache_hit=False,
            passed_filters=True,
        )
        better_candidate = EvalResult(
            fidelity="full64",
            resolution=64,
            compliance=25.0,
            score=25.0,
            volume_fraction=0.57,
            smoothness=110,
            islands=1,
            fea_performed=True,
            cache_hit=False,
            passed_filters=True,
        )
        worse_invalid_candidate = EvalResult(
            fidelity="full64",
            resolution=64,
            compliance=1e12,
            score=1e12,
            volume_fraction=0.01,
            smoothness=4,
            islands=2,
            fea_performed=False,
            cache_hit=False,
            passed_filters=False,
            invalid_reason="volume_out_of_range",
        )

        selected_mask, selected_eval, selection = pipeline_module._select_monotonic_refinement(
            incumbent_mask,
            incumbent,
            candidate_mask,
            better_candidate,
        )
        self.assertTrue(selection["accepted"])
        self.assertTrue(np.array_equal(selected_mask, candidate_mask))
        self.assertEqual(selected_eval.score, better_candidate.score)

        selected_mask, selected_eval, selection = pipeline_module._select_monotonic_refinement(
            incumbent_mask,
            incumbent,
            candidate_mask,
            worse_invalid_candidate,
        )
        self.assertFalse(selection["accepted"])
        self.assertTrue(np.array_equal(selected_mask, incumbent_mask))
        self.assertEqual(selected_eval.score, incumbent.score)
        self.assertEqual(selection["reason"], "rejected_invalid_rl_candidate:volume_out_of_range")

    def test_direct_rl_is_not_skipped_when_direct_budget_is_exhausted(self) -> None:
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=True,
            max_full_evals=12,
            max_rl_full_evals=4,
        )
        fake_artifacts = pipeline_module.DirectSearchArtifacts(
            initial_population=[self.make_mask(64)],
            archive_best=[self.make_mask(64)],
            best64=self.make_mask(64),
            metrics={"seed64": {"score": 1.0}, "best64": {"score": 1.0}},
            fea_counts={"proxy16": 0.0, "proxy32": 0.0, "full64": 12.0, "cache_hits": 0.0, "cache_size": 1.0},
            runtime=1.0,
            warnings=[],
            search_trace=[],
        )

        with patch.object(pipeline_module, "run_direct_search_core", return_value=fake_artifacts), patch.object(
            pipeline_module, "_maybe_run_direct_rl", return_value=(self.make_mask(64), {"accepted_removals": 1})
        ) as mocked_rl:
            artifacts = pipeline_module.run_direct64_exact_search(config)

        mocked_rl.assert_called_once()
        self.assertIn("rl_refined64", artifacts.metrics)
        self.assertIn("rl_refined64_diagnostics", artifacts.metrics)
        self.assertIn("final64", artifacts.metrics)
        self.assertIn("final64_diagnostics", artifacts.metrics)
        self.assertIn("rl_selection", artifacts.metrics)
        self.assertIn("rl_trials", artifacts.metrics)

    def test_direct_rl_invalid_candidate_does_not_overwrite_best64(self) -> None:
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=True,
            max_full_evals=12,
            max_rl_full_evals=4,
        )
        direct_best = self.make_mask(64)
        invalid_candidate = np.zeros((64, 64), dtype=np.uint8)
        fake_artifacts = pipeline_module.DirectSearchArtifacts(
            initial_population=[direct_best],
            archive_best=[direct_best],
            best64=direct_best.copy(),
            metrics={"seed64": {"score": 1.0}, "best64": {"score": 1.0}},
            fea_counts={"proxy16": 0.0, "proxy32": 0.0, "full64": 12.0, "cache_hits": 0.0, "cache_size": 1.0},
            runtime=1.0,
            warnings=[],
            search_trace=[],
        )

        with patch.object(pipeline_module, "run_direct_search_core", return_value=fake_artifacts), patch.object(
            pipeline_module, "_maybe_run_direct_rl", return_value=(invalid_candidate, {"accepted_removals": 0})
        ):
            artifacts = pipeline_module.run_direct64_exact_search(config)

        self.assertTrue(np.array_equal(artifacts.best64, direct_best))
        self.assertFalse(artifacts.metrics["rl_selection"]["accepted"])
        self.assertEqual(artifacts.metrics["final64"]["invalid_reason"], None)
        self.assertEqual(artifacts.metrics["rl_refined64"]["invalid_reason"], "volume_out_of_range")

    def test_direct_rl_runs_from_archive_top_k_seeds(self) -> None:
        config = ProblemConfig(
            resolution=64,
            pipeline_mode="direct64_exact",
            enable_rl=True,
            rl_archive_top_k=2,
            max_full_evals=12,
            max_rl_full_evals=4,
        )
        best = self.make_mask(64)
        second = np.rot90(best)
        third = np.flipud(best)
        fake_artifacts = pipeline_module.DirectSearchArtifacts(
            initial_population=[best],
            archive_best=[best, second, third],
            best64=best.copy(),
            metrics={"seed64": {"score": 1.0}, "best64": {"score": 1.0}},
            fea_counts={"proxy16": 0.0, "proxy32": 0.0, "full64": 12.0, "cache_hits": 0.0, "cache_size": 3.0},
            runtime=1.0,
            warnings=[],
            search_trace=[],
        )

        with patch.object(pipeline_module, "run_direct_search_core", return_value=fake_artifacts), patch.object(
            pipeline_module, "_maybe_run_direct_rl", side_effect=[(best.copy(), {"accepted_removals": 0}), (second.copy(), {"accepted_removals": 1})]
        ) as mocked_rl:
            artifacts = pipeline_module.run_direct64_exact_search(config)

        self.assertEqual(mocked_rl.call_count, 2)
        self.assertEqual(len(artifacts.metrics["rl_trials"]), 2)
        self.assertEqual(artifacts.metrics["rl_selection"]["seed_count"], 2)
        self.assertIn("diagnostics", artifacts.metrics["rl_trials"][0])

    def test_cli_runner_writes_summary_and_masks(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            exit_code = cli_main(
                [
                    "--resolution",
                    "32",
                    "--runtime-budget-hours",
                    "0.01",
                    "--coarse-population",
                    "10",
                    "--coarse-generations",
                    "3",
                    "--coarse-elite-count",
                    "4",
                    "--stage32-top-k",
                    "4",
                    "--stage64-top-k",
                    "2",
                    "--local-search-steps32",
                    "3",
                    "--local-search-steps64",
                    "3",
                    "--max-full-evals",
                    "80",
                    "--output-dir",
                    tmp_dir,
                ]
            )
            self.assertEqual(exit_code, 0)
            summary_path = Path(tmp_dir) / "summary.json"
            coarse_path = Path(tmp_dir) / "coarse16.npy"
            refined32_path = Path(tmp_dir) / "refined32.npy"
            refined64_png_path = Path(tmp_dir) / "refined64.png"
            self.assertTrue(summary_path.exists())
            self.assertTrue(coarse_path.exists())
            self.assertTrue(refined32_path.exists())
            self.assertTrue(refined64_png_path.exists())

    def test_direct_cli_runner_writes_summary_and_best_mask(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            exit_code = cli_main(
                [
                    "--pipeline-mode",
                    "direct64_exact",
                    "--resolution",
                    "64",
                    "--runtime-budget-hours",
                    "0.001",
                    "--direct-population",
                    "4",
                    "--direct-elite-count",
                    "2",
                    "--direct-offspring-batch",
                    "2",
                    "--direct-archive-size",
                    "3",
                    "--direct-restart-stagnation-evals",
                    "12",
                    "--max-full-evals",
                    "12",
                    "--workers",
                    "1",
                    "--output-dir",
                    tmp_dir,
                ]
            )
            self.assertEqual(exit_code, 0)
            summary_path = Path(tmp_dir) / "summary.json"
            seed_path = Path(tmp_dir) / "seed64.npy"
            best_path = Path(tmp_dir) / "best64.npy"
            archive_path = Path(tmp_dir) / "archive.json"
            best_png_path = Path(tmp_dir) / "best64.png"
            self.assertTrue(summary_path.exists())
            self.assertTrue(seed_path.exists())
            self.assertTrue(best_path.exists())
            self.assertTrue(archive_path.exists())
            self.assertTrue(best_png_path.exists())

    def test_save_outputs_serializes_eval_results_inside_direct_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = ProblemConfig(
                resolution=64,
                pipeline_mode="direct64_exact",
                enable_rl=True,
                max_full_evals=12,
                max_rl_full_evals=4,
            )
            evaluation = EvalResult(
                fidelity="full64",
                resolution=64,
                compliance=26.33,
                score=26.33,
                volume_fraction=0.57,
                smoothness=1000,
                islands=1,
                fea_performed=True,
                cache_hit=False,
                passed_filters=True,
            )
            mask = self.make_mask(64)
            artifacts = SimpleNamespace(
                initial_population=[mask.copy()],
                archive_best=[mask.copy()],
                best64=mask.copy(),
                search_trace=[{"event": "rl_refinement_trial", "diagnostics": {"last_info": {"evaluation": evaluation}}}],
                metrics={"rl_trials": [{"diagnostics": {"last_info": {"evaluation": evaluation}}}]},
                warnings=[],
                runtime=1.0,
                fea_counts={"proxy16": 0.0, "proxy32": 0.0, "full64": 1.0, "cache_hits": 0.0, "cache_size": 1.0},
            )

            saved = _save_outputs(Path(tmp_dir), artifacts, config)
            summary_text = Path(saved["summary"]).read_text()
            archive_text = Path(saved["archive"]).read_text()

            self.assertIn('"score": 26.33', summary_text)
            self.assertIn('"score": 26.33', archive_text)


if __name__ == "__main__":
    unittest.main()
