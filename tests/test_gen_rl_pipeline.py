from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gigala.topology.topology_optimiz.gen_rl import ProblemConfig, run_multistage_search
from gigala.topology.topology_optimiz.gen_rl.cli import main as cli_main
from gigala.topology.topology_optimiz.gen_rl.fem import Evaluator
from gigala.topology.topology_optimiz.gen_rl.pipeline import _resolve_rl_device
from gigala.topology.topology_optimiz.gen_rl.refine_env import build_action_catalog, compute_action_mask
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


if __name__ == "__main__":
    unittest.main()
