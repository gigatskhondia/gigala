from __future__ import annotations

import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gigala.topology.topology_optimiz.gen_rl import ProblemConfig, run_direct64_exact_search, run_multistage_search
from gigala.topology.topology_optimiz.gen_rl.cli import main as cli_main
from gigala.topology.topology_optimiz.gen_rl.direct_search import _init_worker, build_mutation_coverage, evaluate_exact_batch
from gigala.topology.topology_optimiz.gen_rl.fem import Evaluator
from gigala.topology.topology_optimiz.gen_rl.pipeline import _resolve_rl_device
from gigala.topology.topology_optimiz.gen_rl.refine_env import (
    build_action_catalog,
    compute_action_mask,
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
        self.assertEqual(observation.shape, (3, 64, 64))
        self.assertIn("evaluation", info)
        action_masks = env.action_masks()
        self.assertGreater(int(action_masks.sum()), 0)
        action = int(np.flatnonzero(action_masks)[0])
        next_observation, _reward, _terminated, _truncated, info = env.step(action)
        self.assertEqual(next_observation.shape, (3, 64, 64))
        self.assertIn("evaluation", info)

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


if __name__ == "__main__":
    unittest.main()
