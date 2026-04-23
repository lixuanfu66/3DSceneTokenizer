from __future__ import annotations

import math
import unittest

import numpy as np

from threedvae.geometry.bbox import estimate_ground_aligned_obb, obb_vertices, quad_faces
from threedvae.geometry.frames import pose_from_obb, transform_points_to_local


class GeometryTest(unittest.TestCase):
    def test_estimate_ground_aligned_obb_for_axis_aligned_box(self) -> None:
        xyz = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 4.0, 1.0],
                [0.0, 4.0, 1.0],
            ],
            dtype=np.float32,
        )
        obb = estimate_ground_aligned_obb(xyz)

        self.assertTrue(np.allclose(obb.center_xyz, np.asarray([1.0, 2.0, 0.5], dtype=np.float32)))
        self.assertTrue(np.allclose(np.sort(obb.size_xyz[:2]), np.asarray([2.0, 4.0], dtype=np.float32)))
        self.assertAlmostEqual(obb.size_xyz[2], 1.0, places=5)

    def test_transform_points_to_local_recenters_points(self) -> None:
        xyz = np.asarray(
            [
                [1.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 3.0, 0.0],
                [1.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        obb = estimate_ground_aligned_obb(xyz)
        pose = pose_from_obb(obb)
        local_xyz = transform_points_to_local(xyz, pose)

        self.assertTrue(np.allclose(local_xyz.mean(axis=0), np.asarray([0.0, 0.0, 0.0], dtype=np.float32), atol=1e-5))

    def test_quad_faces_shape(self) -> None:
        self.assertEqual(quad_faces().shape, (6, 4))

