from __future__ import annotations

import unittest

import numpy as np

from threedvae.data.dataset import build_layered_udf_queries, build_udf_queries


class QuerySamplingTest(unittest.TestCase):
    def test_uniform_udf_queries_have_expected_shape_and_range(self) -> None:
        xyz = _make_points()
        queries, udf = build_udf_queries(
            xyz,
            num_queries=16,
            truncation_distance=0.25,
            seed=1,
        )

        self.assertEqual(queries.shape, (16, 3))
        self.assertEqual(udf.shape, (16, 1))
        self.assertTrue(np.all(udf >= 0.0))
        self.assertTrue(np.all(udf <= 0.25))

    def test_layered_udf_queries_have_expected_shape_and_range(self) -> None:
        xyz = _make_points()
        queries, udf = build_layered_udf_queries(
            xyz,
            node_center=np.asarray([0.5, 0.5, 0.0], dtype=np.float32),
            node_size=np.asarray([1.0, 1.0, 0.2], dtype=np.float32),
            semantic_id=1,
            split_flag=0b011,
            num_queries=32,
            truncation_distance=0.25,
            seed=2,
        )

        self.assertEqual(queries.shape, (32, 3))
        self.assertEqual(udf.shape, (32, 1))
        self.assertTrue(np.all(udf >= 0.0))
        self.assertTrue(np.all(udf <= 0.25))


def _make_points() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.1],
        ],
        dtype=np.float32,
    )


if __name__ == "__main__":
    unittest.main()
