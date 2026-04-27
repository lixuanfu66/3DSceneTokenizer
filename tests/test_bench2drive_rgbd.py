from __future__ import annotations

from pathlib import Path
import shutil
import unittest

import numpy as np

from threedvae.data.bench2drive_rgbd import (
    adjust_depth,
    bench2drive_sensor_key,
    decode_depth,
    decode_label,
    dequantize_depth,
    opencv_camera_to_carla_camera,
    project_depth_to_points,
    transform_points,
    write_point_cloud_ply,
)
from threedvae.data.loaders.ply_loader import load_ply_frame


_TEST_TMP_ROOT = Path("D:/code/3dVAE/.tmp/test_runs")


class Bench2DriveRgbdTest(unittest.TestCase):
    def test_decode_carla_rgb_depth(self) -> None:
        depth_rgb = np.asarray([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)

        depth_m = decode_depth(depth_rgb, "carla-rgb")

        self.assertAlmostEqual(float(depth_m[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(depth_m[0, 1]), 1000.0, places=4)

    def test_project_depth_to_camera_points(self) -> None:
        depth = np.asarray([[2.0, 4.0]], dtype=np.float32)
        intrinsic = np.asarray([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        points, pixels = project_depth_to_points(depth, intrinsic)

        expected_points = np.asarray([[0.0, 0.0, 2.0], [2.0, 0.0, 4.0]], dtype=np.float32)
        expected_pixels = np.asarray([[0, 0], [0, 1]], dtype=np.int64)
        self.assertTrue(np.allclose(points, expected_points))
        self.assertTrue(np.array_equal(pixels, expected_pixels))

    def test_decode_rgb24_label(self) -> None:
        label = np.asarray([[[1, 0, 0], [1, 2, 3]]], dtype=np.uint8)

        decoded = decode_label(label, "rgb24")

        self.assertTrue(np.array_equal(decoded, np.asarray([[1, 197121]], dtype=np.int32)))

    def test_adjust_depth_applies_scale_and_offset(self) -> None:
        depth = np.asarray([[0.0, 255.0]], dtype=np.float32)

        adjusted = adjust_depth(depth, scale=1000.0 / 255.0, offset=1.0)

        self.assertTrue(np.allclose(adjusted, np.asarray([[1.0, 1001.0]], dtype=np.float32)))

    def test_dequantize_depth_spreads_equal_column_runs(self) -> None:
        depth = np.asarray([[10.0, 5.0], [10.0, 6.0], [10.0, 6.0], [11.0, 6.0]], dtype=np.float32)

        dequantized = dequantize_depth(depth, "column-ramp")

        expected = np.asarray(
            [
                [10.0 + 1.0 / 6.0, 5.0],
                [10.5, 6.0 + 1.0 / 6.0],
                [10.0 + 5.0 / 6.0, 6.5],
                [11.0, 6.0 + 5.0 / 6.0],
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(dequantized, expected))

    def test_dequantize_depth_can_reverse_equal_column_runs(self) -> None:
        depth = np.asarray([[10.0], [10.0], [10.0]], dtype=np.float32)

        dequantized = dequantize_depth(depth, "column-ramp-reverse")

        expected = np.asarray([[10.0 + 5.0 / 6.0], [10.5], [10.0 + 1.0 / 6.0]], dtype=np.float32)
        self.assertTrue(np.allclose(dequantized, expected))

    def test_dequantize_depth_auto_reverses_when_column_depth_decreases(self) -> None:
        depth = np.asarray([[11.0], [10.0], [10.0], [10.0], [9.0]], dtype=np.float32)

        dequantized = dequantize_depth(depth, "column-ramp-auto")

        expected = np.asarray([[11.0], [10.0 + 5.0 / 6.0], [10.5], [10.0 + 1.0 / 6.0], [9.0]], dtype=np.float32)
        self.assertTrue(np.allclose(dequantized, expected))

    def test_dequantize_depth_can_split_runs_by_semantic(self) -> None:
        depth = np.asarray([[10.0], [10.0], [10.0], [10.0]], dtype=np.float32)
        semantic = np.asarray([[1], [1], [2], [2]], dtype=np.int32)

        dequantized = dequantize_depth(depth, "column-ramp", semantic=semantic, semantic_aware=True)

        expected = np.asarray([[10.25], [10.75], [10.25], [10.75]], dtype=np.float32)
        self.assertTrue(np.allclose(dequantized, expected))

    def test_dequantize_depth_min_run_leaves_short_runs_unchanged(self) -> None:
        depth = np.asarray([[10.0], [10.0], [10.0]], dtype=np.float32)

        dequantized = dequantize_depth(depth, "column-ramp", min_run=4)

        self.assertTrue(np.array_equal(dequantized, depth))

    def test_dequantize_depth_can_limit_to_included_semantics(self) -> None:
        depth = np.asarray([[10.0], [10.0], [10.0], [10.0]], dtype=np.float32)
        semantic = np.asarray([[1], [24], [2], [2]], dtype=np.int32)

        dequantized = dequantize_depth(depth, "column-ramp", semantic=semantic, include_semantics={1, 24})

        expected = np.asarray([[10.25], [10.75], [10.0], [10.0]], dtype=np.float32)
        self.assertTrue(np.allclose(dequantized, expected))

    def test_transform_points_uses_homogeneous_matrix(self) -> None:
        points = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)

        transformed = transform_points(points, transform)

        self.assertTrue(np.allclose(transformed, np.asarray([[11.0, 22.0, 33.0]], dtype=np.float32)))

    def test_opencv_camera_to_carla_camera_maps_axes(self) -> None:
        optical_points = np.asarray([[1.0, 2.0, 10.0]], dtype=np.float32)

        carla_points = opencv_camera_to_carla_camera(optical_points)

        self.assertTrue(np.allclose(carla_points, np.asarray([[10.0, 1.0, -2.0]], dtype=np.float32)))

    def test_write_point_cloud_ply_is_compatible_with_loader(self) -> None:
        tmp_dir = _fresh_tmp_dir("bench2drive_rgbd")
        try:
            ply_path = tmp_dir / "frame.ply"
            write_point_cloud_ply(
                ply_path,
                np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.asarray([[4, 5, 6]], dtype=np.uint8),
                np.asarray([7], dtype=np.int32),
                np.asarray([8], dtype=np.int32),
            )

            frame = load_ply_frame(str(ply_path))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertTrue(np.allclose(frame.xyz, np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)))
        self.assertTrue(np.array_equal(frame.rgb, np.asarray([[4, 5, 6]], dtype=np.uint8)))
        self.assertTrue(np.array_equal(frame.instance_id, np.asarray([7], dtype=np.int32)))
        self.assertTrue(np.array_equal(frame.semantic_id, np.asarray([8], dtype=np.int32)))

    def test_camera_name_maps_to_bench2drive_sensor_key(self) -> None:
        self.assertEqual(bench2drive_sensor_key("front_left"), "CAM_FRONT_LEFT")
        self.assertEqual(bench2drive_sensor_key("CAM_BACK"), "CAM_BACK")


def _fresh_tmp_dir(name: str) -> Path:
    path = _TEST_TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path
