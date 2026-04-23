from __future__ import annotations

import shutil
import textwrap
import unittest
from pathlib import Path

from threedvae.data.dataset import build_instance_dataset_from_ply_dir, collect_instance_samples_from_ply_paths


_TEST_TMP_ROOT = Path("D:/code/3dVAE/.tmp/test_runs")


class DatasetTest(unittest.TestCase):
    def test_collect_instance_samples_from_ply_paths(self) -> None:
        tmp_dir = _fresh_tmp_dir("dataset_collect")
        try:
            ply_path = tmp_dir / "frame_0001.ply"
            ply_path.write_text(_sample_ply(), encoding="utf-8")
            samples = collect_instance_samples_from_ply_paths([str(ply_path)])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].xyz_local.shape[1], 3)

    def test_build_instance_dataset_from_ply_dir(self) -> None:
        tmp_dir = _fresh_tmp_dir("dataset_dir")
        try:
            ply_path = tmp_dir / "frame_0001.ply"
            ply_path.write_text(_sample_ply(), encoding="utf-8")
            dataset = build_instance_dataset_from_ply_dir(str(tmp_dir), points_per_instance=4)
            sample = dataset[0]
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(sample["points"].shape, (4, 6))
        self.assertEqual(sample["xyz"].shape, (4, 3))
        self.assertEqual(sample["rgb"].shape, (4, 3))


def _sample_ply() -> str:
    return textwrap.dedent(
        """\
        ply
        format ascii 1.0
        element vertex 6
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
        3 0 0 0 255 0 2 2
        4 0 0 0 255 0 2 2
        4 1 0 0 255 0 2 2
        """
    )


def _fresh_tmp_dir(name: str) -> Path:
    path = _TEST_TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path

