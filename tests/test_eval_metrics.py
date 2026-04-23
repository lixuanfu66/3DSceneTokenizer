from __future__ import annotations

import unittest

import numpy as np

from threedvae.eval.metrics import chamfer_l2_numpy, summarize_codebook_usage, summarize_compression


class EvalMetricsTest(unittest.TestCase):
    def test_chamfer_l2_numpy_is_zero_for_identical_sets(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.assertAlmostEqual(chamfer_l2_numpy(points, points), 0.0, places=6)

    def test_summarize_codebook_usage(self) -> None:
        summary = summarize_codebook_usage([0, 0, 2, 2, 2, 3], codebook_size=8)
        self.assertEqual(summary.code_count, 8)
        self.assertEqual(summary.used_code_count, 3)
        self.assertGreater(summary.usage_rate, 0.0)
        self.assertGreater(summary.entropy_bits, 0.0)
        self.assertGreater(summary.perplexity, 1.0)

    def test_summarize_compression(self) -> None:
        summary = summarize_compression([10, 20, 30], sample_unit="node")
        self.assertEqual(summary.sample_count, 3)
        self.assertAlmostEqual(summary.avg_input_points, 20.0, places=6)
        self.assertAlmostEqual(summary.latent_tokens_per_sample, 1.0, places=6)
        self.assertAlmostEqual(summary.points_per_token_ratio, 20.0, places=6)
