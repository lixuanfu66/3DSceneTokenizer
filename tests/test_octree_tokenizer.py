from __future__ import annotations

import unittest

import numpy as np

from threedvae.data.schema import GroundAlignedOBB, Pose3DYaw, SceneInstance
from threedvae.octree.split_policy import OctreeBuildConfig, SemanticOctreePolicy
from threedvae.octree.tree import build_instance_octree, build_octree_debug_records
from threedvae.tokenizer.instance_encoder import encode_instance


class OctreeTokenizerTest(unittest.TestCase):
    def test_build_instance_octree_splits_dense_high_priority_instance(self) -> None:
        instance = _make_dense_instance()
        config = OctreeBuildConfig(
            high_priority_semantics={1},
            min_points_per_node=4,
            geom_threshold=0.2,
            rgb_threshold=0.05,
        )

        nodes = build_instance_octree(instance, config)

        self.assertGreater(len(nodes), 1)
        self.assertEqual(nodes[0].structure_state, "split")
        self.assertEqual(nodes[0].split_flag, 0b111)
        self.assertNotEqual(nodes[0].child_mask, 0)

    def test_encode_instance_returns_tokens_and_debug_records(self) -> None:
        instance = _make_dense_instance()
        config = OctreeBuildConfig(
            high_priority_semantics={1},
            min_points_per_node=4,
            geom_threshold=0.2,
            rgb_threshold=0.05,
        )

        result = encode_instance(instance, octree_config=config)
        debug_records = build_octree_debug_records(instance, result.octree_nodes)

        self.assertEqual(len(result.token_block.tokens), len(result.octree_nodes))
        self.assertGreaterEqual(result.token_block.max_depth, 1)
        self.assertEqual(debug_records[0].vertices_local.shape, (8, 3))
        self.assertEqual(debug_records[0].quad_faces.shape, (6, 4))
        self.assertEqual(result.token_block.tokens[0].split_flag, 0b111)
        self.assertEqual(result.token_block.tokens[0].path_code, "root")
        self.assertEqual(result.token_block.tokens[0].visibility_state, "observed_surface")
        if len(result.token_block.tokens) > 1:
            self.assertIsNotNone(result.token_block.tokens[1].parent_id)
            self.assertIn(".", result.token_block.tokens[1].path_code)

    def test_encode_instance_can_attach_learned_code(self) -> None:
        instance = _make_dense_instance()
        result = encode_instance(instance, node_code_provider=_StubNodeCodeProvider())

        self.assertTrue(all(token.learned_code is not None for token in result.token_block.tokens))
        self.assertTrue(all(token.code_source == "learned_vq+rule" for token in result.token_block.tokens))

    def test_build_instance_octree_can_degenerate_to_xy_split(self) -> None:
        instance = _make_ground_like_instance()
        config = OctreeBuildConfig(
            min_points_per_node=2,
            geom_threshold=0.1,
            rgb_threshold=0.01,
        )

        nodes = build_instance_octree(instance, config)
        root = nodes[0]
        level_one_children = [node for node in nodes if node.parent_id == root.node_id]

        self.assertEqual(root.split_flag, 0b011)
        self.assertEqual(root.structure_state, "split")
        self.assertLess(root.child_mask, 1 << 4)
        self.assertLessEqual(len(level_one_children), 4)
        self.assertTrue(all(child.child_index in {0, 1, 2, 3} for child in level_one_children))

    def test_planar_instance_split_is_driven_by_geometry_without_semantic_override(self) -> None:
        instance = _make_ground_like_instance()
        config = OctreeBuildConfig(
            min_points_per_node=2,
            geom_threshold=0.3,
            rgb_threshold=1.0,
        )

        nodes = build_instance_octree(instance, config)
        root = nodes[0]

        self.assertEqual(root.split_flag, 0b011)
        self.assertEqual(root.structure_state, "split")
        self.assertGreater(root.geom_score, config.geom_threshold)

    def test_semantic_policy_can_control_depth_by_distance(self) -> None:
        near_instance = _make_vehicle_instance(center_xyz=np.asarray([10.0, 0.0, 0.0], dtype=np.float32))
        far_instance = _make_vehicle_instance(center_xyz=np.asarray([50.0, 0.0, 0.0], dtype=np.float32))
        config = OctreeBuildConfig(
            semantic_policies={
                14: SemanticOctreePolicy(
                    tag_name="Car",
                    min_depth=1,
                    max_depth_by_distance={"near": 4, "mid": 3, "far": 1},
                    preferred_split_flag=0b111,
                    lock_preferred_split=False,
                    priority="high",
                )
            }
        )

        self.assertEqual(config.max_depth_for(14, near_instance.pose_ego.center_xyz), 4)
        self.assertEqual(config.max_depth_for(14, far_instance.pose_ego.center_xyz), 1)

    def test_semantic_policy_can_lock_preferred_split_flag(self) -> None:
        instance = _make_ground_like_instance()
        config = OctreeBuildConfig(
            semantic_policies={
                99: SemanticOctreePolicy(
                    tag_name="GroundLike",
                    min_depth=1,
                    max_depth_by_distance={"near": 3, "mid": 2, "far": 1},
                    preferred_split_flag=0b011,
                    lock_preferred_split=True,
                    priority="medium",
                )
            },
            min_points_per_node=2,
            geom_threshold=0.1,
            rgb_threshold=0.01,
        )

        nodes = build_instance_octree(instance, config)
        self.assertEqual(nodes[0].split_flag, 0b011)


class _StubNodeCodeProvider:
    def encode_node(self, xyz_local: np.ndarray, rgb: np.ndarray) -> int:
        return int(xyz_local.shape[0] * 1000 + int(np.mean(rgb[:, 0])))


def _make_dense_instance() -> SceneInstance:
    grid = np.asarray(
        [
            [-1.0, -1.0, -0.5],
            [1.0, -1.0, -0.5],
            [1.0, 1.0, -0.5],
            [-1.0, 1.0, -0.5],
            [-1.0, -1.0, 0.5],
            [1.0, -1.0, 0.5],
            [1.0, 1.0, 0.5],
            [-1.0, 1.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    rgb = np.asarray(
        [
            [255, 0, 0],
            [250, 10, 0],
            [245, 20, 0],
            [240, 30, 0],
            [235, 40, 0],
            [230, 50, 0],
            [225, 60, 0],
            [220, 70, 0],
            [100, 100, 100],
            [110, 110, 110],
            [120, 120, 120],
            [130, 130, 130],
            [140, 140, 140],
        ],
        dtype=np.uint8,
    )
    bbox = GroundAlignedOBB(
        center_xyz=np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
        size_xyz=np.asarray([2.0, 2.0, 1.0], dtype=np.float32),
        yaw=0.0,
    )
    return SceneInstance(
        instance_id=1,
        semantic_id=1,
        category_name="semantic_1",
        xyz_ego=grid + np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
        rgb=rgb,
        bbox_ego=bbox,
        pose_ego=Pose3DYaw(center_xyz=bbox.center_xyz.copy(), yaw=bbox.yaw),
        xyz_local=grid.copy(),
        num_points=grid.shape[0],
    )


def _make_ground_like_instance() -> SceneInstance:
    grid = np.asarray(
        [
            [-4.0, -4.0, 0.00],
            [4.0, -4.0, 0.00],
            [4.0, 4.0, 0.01],
            [-4.0, 4.0, 0.01],
            [0.0, 0.0, 0.00],
            [2.0, 2.0, 0.01],
            [-2.0, 2.0, 0.00],
            [2.0, -2.0, 0.00],
        ],
        dtype=np.float32,
    )
    rgb = np.asarray(
        [
            [100, 100, 100],
            [102, 102, 102],
            [104, 104, 104],
            [106, 106, 106],
            [108, 108, 108],
            [110, 110, 110],
            [112, 112, 112],
            [114, 114, 114],
        ],
        dtype=np.uint8,
    )
    bbox = GroundAlignedOBB(
        center_xyz=np.asarray([0.0, 0.0, 0.005], dtype=np.float32),
        size_xyz=np.asarray([8.0, 8.0, 0.02], dtype=np.float32),
        yaw=0.0,
    )
    return SceneInstance(
        instance_id=99,
        semantic_id=99,
        category_name="ground",
        xyz_ego=grid.copy(),
        rgb=rgb,
        bbox_ego=bbox,
        pose_ego=Pose3DYaw(center_xyz=bbox.center_xyz.copy(), yaw=bbox.yaw),
        xyz_local=grid.copy(),
        num_points=grid.shape[0],
    )


def _make_vehicle_instance(center_xyz: np.ndarray) -> SceneInstance:
    local = np.asarray(
        [
            [-1.0, -0.8, -0.6],
            [1.0, -0.8, -0.6],
            [1.0, 0.8, -0.6],
            [-1.0, 0.8, -0.6],
            [-1.0, -0.8, 0.6],
            [1.0, -0.8, 0.6],
            [1.0, 0.8, 0.6],
            [-1.0, 0.8, 0.6],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rgb = np.full((local.shape[0], 3), 180, dtype=np.uint8)
    bbox = GroundAlignedOBB(
        center_xyz=center_xyz.astype(np.float32, copy=True),
        size_xyz=np.asarray([2.0, 1.6, 1.2], dtype=np.float32),
        yaw=0.0,
    )
    return SceneInstance(
        instance_id=14,
        semantic_id=14,
        category_name="car",
        xyz_ego=local + center_xyz.astype(np.float32, copy=False),
        rgb=rgb,
        bbox_ego=bbox,
        pose_ego=Pose3DYaw(center_xyz=bbox.center_xyz.copy(), yaw=bbox.yaw),
        xyz_local=local.copy(),
        num_points=local.shape[0],
    )
