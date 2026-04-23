from __future__ import annotations

import json
import shutil
import textwrap
import unittest
from pathlib import Path

from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.pipelines.build_scene_tokens import run_single_frame_pipeline


_TEST_TMP_ROOT = Path("D:/code/3dVAE/.tmp/test_runs")


class PipelineTest(unittest.TestCase):
    def test_run_single_frame_pipeline_writes_packet_and_debug_files(self) -> None:
        tmp_dir = _fresh_tmp_dir("pipeline")
        try:
            ply_path = tmp_dir / "frame_0001.ply"
            ply_path.write_text(_sample_ply(), encoding="utf-8")

            output_dir = tmp_dir / "outputs"
            packet_path = run_single_frame_pipeline(
                str(ply_path),
                str(output_dir),
                scene_id="scene_a",
                frame_id="frame_0001",
                octree_config=OctreeBuildConfig(high_priority_semantics={1}, min_points_per_node=2),
            )

            payload = json.loads(packet_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["scene_id"], "scene_a")
            self.assertEqual(payload["frame_id"], "frame_0001")
            self.assertEqual(payload["instance_blocks"][0]["structure_type"], "adaptive_split_tree")
            self.assertEqual(payload["instance_blocks"][0]["tokens"][0]["path_code"], "root")
            self.assertIn("child_mask", payload["instance_blocks"][0]["tokens"][0])
            self.assertEqual(payload["instance_blocks"][0]["tokens"][0]["visibility_state"], "observed_surface")
            self.assertTrue((output_dir / "scene_instance_bboxes.ply").exists())
            self.assertTrue((output_dir / "instance_1_bbox.json").exists())
            self.assertTrue((output_dir / "instance_1_octree_nodes.jsonl").exists())
            self.assertTrue((output_dir / "scene_tokens_compact.json").exists())
            self.assertTrue((output_dir / "scene_tokens_compact_v2.json").exists())
            self.assertTrue((output_dir / "scene_tokens_compact_latest.json").exists())
            self.assertTrue((output_dir / "scene_tokens_llm_sequence.json").exists())
            self.assertTrue((output_dir / "scene_tokens_llm_sequence_v2.json").exists())
            self.assertTrue((output_dir / "scene_tokens_llm_sequence_latest.json").exists())
            self.assertTrue((output_dir / "scene_token_bundle.json").exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_run_single_frame_pipeline_can_emit_learned_codes(self) -> None:
        tmp_dir = _fresh_tmp_dir("pipeline_learned")
        try:
            ply_path = tmp_dir / "frame_0001.ply"
            ply_path.write_text(_sample_ply(), encoding="utf-8")

            output_dir = tmp_dir / "outputs"
            packet_path = run_single_frame_pipeline(
                str(ply_path),
                str(output_dir),
                node_code_provider=_StubNodeCodeProvider(),
            )

            payload = json.loads(packet_path.read_text(encoding="utf-8"))
            token = payload["instance_blocks"][0]["tokens"][0]
            self.assertIsNotNone(token["learned_code"])
            self.assertEqual(token["code_source"], "learned_vq+rule")

            compact_payload = json.loads((output_dir / "scene_tokens_compact.json").read_text(encoding="utf-8"))
            compact_token = compact_payload["instance_blocks"][0]["tokens"][0]
            self.assertNotIn("path_code", compact_token)
            self.assertIn("child_mask", compact_token)
            self.assertEqual(compact_payload["instance_blocks"][0]["traversal_order"], "preorder_dfs")

            compact_v2_payload = json.loads((output_dir / "scene_tokens_compact_v2.json").read_text(encoding="utf-8"))
            compact_v2_block = compact_v2_payload["instance_blocks"][0]
            compact_v2_token = compact_v2_block["tokens"][0]
            self.assertIn("header", compact_v2_block)
            self.assertNotIn("level", compact_v2_token)
            self.assertNotIn("num_points", compact_v2_token)
            self.assertIn("main_code", compact_v2_token)

            sequence_payload = json.loads((output_dir / "scene_tokens_llm_sequence.json").read_text(encoding="utf-8"))
            self.assertEqual(sequence_payload["sequence_format"], "scene_pose_then_instance_preorder_v1")
            self.assertEqual(sequence_payload["tokens"][0]["token_type"], "SCENE_START")
            self.assertEqual(sequence_payload["tokens"][-1]["token_type"], "SCENE_END")
            self.assertTrue(any(token["token_type"] == "INSTANCE_NODE" for token in sequence_payload["tokens"]))

            sequence_v2_payload = json.loads((output_dir / "scene_tokens_llm_sequence_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(sequence_v2_payload["sequence_format"], "instance_header_then_preorder_nodes_v2")
            self.assertEqual(sequence_v2_payload["tokens"][0]["token_type"], "SCENE_START")
            self.assertEqual(sequence_v2_payload["tokens"][1]["token_type"], "INSTANCE_HEADER")
            self.assertEqual(sequence_v2_payload["tokens"][-1]["token_type"], "SCENE_END")
            self.assertFalse(any(token["token_type"] == "POSE" for token in sequence_v2_payload["tokens"]))
            self.assertTrue(any(token["token_type"] == "INSTANCE_NODE_V2" for token in sequence_v2_payload["tokens"]))

            latest_sequence_payload = json.loads((output_dir / "scene_tokens_llm_sequence_latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest_sequence_payload["sequence_format"], "instance_header_then_preorder_nodes_v2")

            bundle_manifest = json.loads((output_dir / "scene_token_bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(bundle_manifest["recommended_llm_sequence"], "scene_tokens_llm_sequence_latest.json")
            self.assertEqual(bundle_manifest["recommended_compact_packet"], "scene_tokens_compact_latest.json")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class _StubNodeCodeProvider:
    def encode_node(self, xyz_local, rgb):
        return int(xyz_local.shape[0] * 10 + rgb.shape[0])


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
