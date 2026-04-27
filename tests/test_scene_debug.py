from __future__ import annotations

import json
import unittest
from pathlib import Path
import shutil

import numpy as np

from threedvae.data.schema import GroundAlignedOBB, Pose3DYaw, SceneFrame, SceneInstance
from threedvae.debug.exporters import (
    export_instance_bbox_json,
    export_instance_points_with_octree_node_bboxes_ply,
    export_scene_instance_bboxes_ply,
)
from threedvae.octree.tree import build_instance_octree, build_octree_debug_records


_TEST_TMP_ROOT = Path("D:/code/3dVAE/.tmp/test_runs")


class SceneDebugExporterTest(unittest.TestCase):
    def test_export_scene_instance_bboxes_ply_writes_faces(self) -> None:
        instance = _make_instance(instance_id=7, semantic_id=3)
        scene = SceneFrame(
            scene_id="scene_a",
            frame_id="frame_0001",
            ply_path="dummy.ply",
            instances=[instance],
            num_points=instance.num_points,
            num_instances=1,
        )

        tmp_dir = _fresh_tmp_dir("scene_bbox_ply")
        try:
            output_path = tmp_dir / "scene_instance_bboxes.ply"
            export_scene_instance_bboxes_ply(scene, str(output_path))
            text = output_path.read_text(encoding="utf-8")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertIn("element vertex 8", text)
        self.assertIn("element face 6", text)
        self.assertIn("property list uchar int vertex_indices", text)

    def test_export_instance_bbox_json_contains_vertices_and_faces(self) -> None:
        instance = _make_instance(instance_id=11, semantic_id=5)

        tmp_dir = _fresh_tmp_dir("instance_bbox_json")
        try:
            output_path = tmp_dir / "instance_11_bbox.json"
            export_instance_bbox_json(instance, str(output_path))
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(len(payload["vertices_ego"]), 8)
        self.assertEqual(len(payload["quad_faces"]), 6)

    def test_export_instance_points_with_octree_node_bboxes_ply_writes_points_and_faces(self) -> None:
        instance = _make_instance(instance_id=17, semantic_id=9)
        nodes = build_instance_octree(instance)
        debug_records = build_octree_debug_records(instance, nodes)

        tmp_dir = _fresh_tmp_dir("instance_points_with_octree_node_bbox_ply")
        try:
            output_path = tmp_dir / "instance_17_points_and_octree_node_bboxes.ply"
            export_instance_points_with_octree_node_bboxes_ply(instance, debug_records, str(output_path))
            text = output_path.read_text(encoding="utf-8")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertIn("element vertex 10", text)
        self.assertIn("element face 6", text)
        self.assertIn("property int vertex_kind", text)
        self.assertIn("comment vertex_kind 0=source_point 1=node_bbox_vertex", text)


def _make_instance(instance_id: int, semantic_id: int) -> SceneInstance:
    bbox = GroundAlignedOBB(
        center_xyz=np.asarray([0.0, 0.0, 0.5], dtype=np.float32),
        size_xyz=np.asarray([2.0, 4.0, 1.0], dtype=np.float32),
        yaw=0.0,
    )
    xyz = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    rgb = np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    return SceneInstance(
        instance_id=instance_id,
        semantic_id=semantic_id,
        category_name=f"semantic_{semantic_id}",
        xyz_ego=xyz,
        rgb=rgb,
        bbox_ego=bbox,
        pose_ego=Pose3DYaw(center_xyz=bbox.center_xyz.copy(), yaw=bbox.yaw),
        xyz_local=xyz.copy(),
        num_points=xyz.shape[0],
    )


def _fresh_tmp_dir(name: str) -> Path:
    path = _TEST_TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path
