from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
import shutil

import numpy as np

from threedvae.data.loaders.ply_loader import load_ply_frame


_TEST_TMP_ROOT = Path("D:/code/3dVAE/.tmp/test_runs")


class PlyLoaderTest(unittest.TestCase):
    def test_load_ascii_ply(self) -> None:
        content = textwrap.dedent(
            """\
            ply
            format ascii 1.0
            element vertex 2
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property int instance
            property int semantic
            end_header
            0 0 0 255 0 0 1 10
            1 0 0 0 255 0 1 10
            """
        )

        tmp_dir = _fresh_tmp_dir("ply_loader")
        try:
            ply_path = tmp_dir / "frame_0001.ply"
            ply_path.write_text(content, encoding="utf-8")

            frame = load_ply_frame(str(ply_path), scene_id="scene_a")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(frame.scene_id, "scene_a")
        self.assertEqual(frame.frame_id, "frame_0001")
        self.assertEqual(frame.xyz.shape, (2, 3))
        self.assertTrue(np.array_equal(frame.instance_id, np.asarray([1, 1], dtype=np.int32)))
        self.assertTrue(np.array_equal(frame.semantic_id, np.asarray([10, 10], dtype=np.int32)))


def _fresh_tmp_dir(name: str) -> Path:
    path = _TEST_TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path
