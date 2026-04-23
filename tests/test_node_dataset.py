from __future__ import annotations

import shutil
import textwrap
import unittest
from pathlib import Path

from threedvae.data.dataset import build_node_dataset_from_ply_dir, collect_node_samples_from_ply_paths
from threedvae.octree.split_policy import OctreeBuildConfig


_TEST_TMP_ROOT = Path("D:/code/3dVAE/.tmp/test_runs")


class NodeDatasetTest(unittest.TestCase):
    def test_collect_node_samples_from_ply_paths(self) -> None:
        tmp_dir = _fresh_tmp_dir("node_dataset_collect")
        try:
            ply_path = tmp_dir / "frame_0001.ply"
            ply_path.write_text(_sample_ply(), encoding="utf-8")
            samples = collect_node_samples_from_ply_paths(
                [str(ply_path)],
                octree_config=OctreeBuildConfig(high_priority_semantics={1}, min_points_per_node=2),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertGreater(len(samples), 1)
        self.assertEqual(samples[0].path_code, "root")
        self.assertIn(samples[0].split_flag, {0b111, 0b011, 0b101, 0b110})

    def test_build_node_dataset_from_ply_dir(self) -> None:
        tmp_dir = _fresh_tmp_dir("node_dataset_dir")
        try:
            ply_path = tmp_dir / "frame_0001.ply"
            ply_path.write_text(_sample_ply(), encoding="utf-8")
            dataset = build_node_dataset_from_ply_dir(
                str(tmp_dir),
                points_per_node=4,
                octree_config=OctreeBuildConfig(high_priority_semantics={1}, min_points_per_node=2),
            )
            sample = dataset[0]
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(sample["sample_type"], "node")
        self.assertEqual(sample["points"].shape, (4, 6))
        self.assertIn("path_code", sample)
        self.assertIn("split_flag", sample)
        self.assertEqual(sample["query_xyz"].shape, (64, 3))
        self.assertEqual(sample["query_udf"].shape, (64, 1))
        self.assertGreaterEqual(float(sample["query_udf"].min()), 0.0)


def _sample_ply() -> str:
    return textwrap.dedent(
        """\
        ply
        format ascii 1.0
        element vertex 8
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property int instance
        property int semantic
        end_header
        0 0 0 255 0 0 1 1
        1 0 0 255 0 0 1 1
        1 1 0 255 0 0 1 1
        0 1 0 255 0 0 1 1
        0 0 1 0 255 0 1 1
        1 0 1 0 255 0 1 1
        1 1 1 0 255 0 1 1
        0 1 1 0 255 0 1 1
        """
    )


def _fresh_tmp_dir(name: str) -> Path:
    path = _TEST_TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path
