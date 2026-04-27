from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from threedvae.data.loaders.ply_loader import load_ply_frame
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.octree.tree import OctreeNode, build_instance_octree
from threedvae.scene.builder import build_scene_frame
from threedvae.utils.torch_compat import HAS_TORCH, require_torch


@dataclass(slots=True)
class InstancePointCloudSample:
    scene_id: str
    frame_id: str
    instance_id: int
    semantic_id: int
    xyz_local: np.ndarray
    rgb: np.ndarray


@dataclass(slots=True)
class TreeNodePointCloudSample:
    scene_id: str
    frame_id: str
    instance_id: int
    semantic_id: int
    node_id: int
    parent_id: int | None
    child_index: int | None
    path_code: str
    level: int
    split_flag: int
    center_local: np.ndarray
    size_local: np.ndarray
    xyz_local: np.ndarray
    rgb: np.ndarray


class InstancePointCloudDataset:
    def __init__(
        self,
        samples: list[InstancePointCloudSample],
        *,
        points_per_instance: int = 256,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.points_per_instance = int(points_per_instance)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        xyz, rgb = sample_point_cloud(
            sample.xyz_local,
            sample.rgb,
            num_points=self.points_per_instance,
            seed=self.seed + index,
        )

        features = np.concatenate([xyz, rgb.astype(np.float32) / 255.0], axis=1).astype(np.float32)
        return {
            "scene_id": sample.scene_id,
            "frame_id": sample.frame_id,
            "instance_id": sample.instance_id,
            "semantic_id": sample.semantic_id,
            "sample_type": "instance",
            "xyz": xyz.astype(np.float32),
            "rgb": rgb.astype(np.float32) / 255.0,
            "points": features,
        }

    def torch_collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        torch, _, _, _, _ = require_torch()
        return {
            "scene_id": [item["scene_id"] for item in batch],
            "frame_id": [item["frame_id"] for item in batch],
            "sample_type": [item["sample_type"] for item in batch],
            "instance_id": torch.as_tensor([item["instance_id"] for item in batch], dtype=torch.int64),
            "semantic_id": torch.as_tensor([item["semantic_id"] for item in batch], dtype=torch.int64),
            "xyz": torch.as_tensor(np.stack([item["xyz"] for item in batch], axis=0), dtype=torch.float32),
            "rgb": torch.as_tensor(np.stack([item["rgb"] for item in batch], axis=0), dtype=torch.float32),
            "points": torch.as_tensor(np.stack([item["points"] for item in batch], axis=0), dtype=torch.float32),
        }


class TreeNodePointCloudDataset:
    def __init__(
        self,
        samples: list[TreeNodePointCloudSample],
        *,
        points_per_node: int = 128,
        queries_per_node: int = 64,
        udf_truncation: float = 0.25,
        near_surface_threshold: float | None = None,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.points_per_node = int(points_per_node)
        self.queries_per_node = int(queries_per_node)
        self.udf_truncation = float(udf_truncation)
        self.near_surface_threshold = (
            min(0.03, 0.2 * self.udf_truncation)
            if near_surface_threshold is None
            else float(near_surface_threshold)
        )
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        xyz, rgb = sample_point_cloud(
            sample.xyz_local,
            sample.rgb,
            num_points=self.points_per_node,
            seed=self.seed + index,
        )
        query_xyz, query_udf = build_udf_queries(
            sample.xyz_local,
            num_queries=self.queries_per_node,
            truncation_distance=self.udf_truncation,
            seed=self.seed + 100_000 + index,
        )
        query_occ = (query_udf <= self.near_surface_threshold).astype(np.float32)

        features = np.concatenate([xyz, rgb.astype(np.float32) / 255.0], axis=1).astype(np.float32)
        return {
            "scene_id": sample.scene_id,
            "frame_id": sample.frame_id,
            "instance_id": sample.instance_id,
            "semantic_id": sample.semantic_id,
            "sample_type": "node",
            "node_id": sample.node_id,
            "parent_id": -1 if sample.parent_id is None else sample.parent_id,
            "child_index": -1 if sample.child_index is None else sample.child_index,
            "path_code": sample.path_code,
            "level": sample.level,
            "split_flag": sample.split_flag,
            "node_center_local": sample.center_local.astype(np.float32, copy=False),
            "node_size_local": sample.size_local.astype(np.float32, copy=False),
            "xyz": xyz.astype(np.float32),
            "rgb": rgb.astype(np.float32) / 255.0,
            "points": features,
            "query_xyz": query_xyz.astype(np.float32),
            "query_udf": query_udf.astype(np.float32),
            "query_occ": query_occ.astype(np.float32),
            "query_rgb_xyz": xyz.astype(np.float32),
            "query_rgb": rgb.astype(np.float32) / 255.0,
            "query_rgb_mask": np.ones((xyz.shape[0], 1), dtype=np.float32),
        }

    def torch_collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        torch, _, _, _, _ = require_torch()
        return {
            "scene_id": [item["scene_id"] for item in batch],
            "frame_id": [item["frame_id"] for item in batch],
            "sample_type": [item["sample_type"] for item in batch],
            "instance_id": torch.as_tensor([item["instance_id"] for item in batch], dtype=torch.int64),
            "semantic_id": torch.as_tensor([item["semantic_id"] for item in batch], dtype=torch.int64),
            "node_id": torch.as_tensor([item["node_id"] for item in batch], dtype=torch.int64),
            "parent_id": torch.as_tensor([item["parent_id"] for item in batch], dtype=torch.int64),
            "child_index": torch.as_tensor([item["child_index"] for item in batch], dtype=torch.int64),
            "path_code": [item["path_code"] for item in batch],
            "level": torch.as_tensor([item["level"] for item in batch], dtype=torch.int64),
            "split_flag": torch.as_tensor([item["split_flag"] for item in batch], dtype=torch.int64),
            "node_center_local": torch.as_tensor(np.stack([item["node_center_local"] for item in batch], axis=0), dtype=torch.float32),
            "node_size_local": torch.as_tensor(np.stack([item["node_size_local"] for item in batch], axis=0), dtype=torch.float32),
            "xyz": torch.as_tensor(np.stack([item["xyz"] for item in batch], axis=0), dtype=torch.float32),
            "rgb": torch.as_tensor(np.stack([item["rgb"] for item in batch], axis=0), dtype=torch.float32),
            "points": torch.as_tensor(np.stack([item["points"] for item in batch], axis=0), dtype=torch.float32),
            "query_xyz": torch.as_tensor(np.stack([item["query_xyz"] for item in batch], axis=0), dtype=torch.float32),
            "query_udf": torch.as_tensor(np.stack([item["query_udf"] for item in batch], axis=0), dtype=torch.float32),
            "query_occ": torch.as_tensor(np.stack([item["query_occ"] for item in batch], axis=0), dtype=torch.float32),
            "query_rgb_xyz": torch.as_tensor(np.stack([item["query_rgb_xyz"] for item in batch], axis=0), dtype=torch.float32),
            "query_rgb": torch.as_tensor(np.stack([item["query_rgb"] for item in batch], axis=0), dtype=torch.float32),
            "query_rgb_mask": torch.as_tensor(np.stack([item["query_rgb_mask"] for item in batch], axis=0), dtype=torch.float32),
        }


def collect_instance_samples_from_ply_paths(
    ply_paths: list[str],
    *,
    drop_negative_instance_ids: bool = True,
) -> list[InstancePointCloudSample]:
    samples: list[InstancePointCloudSample] = []
    for path in ply_paths:
        frame = load_ply_frame(path)
        scene = build_scene_frame(frame, drop_negative_instance_ids=drop_negative_instance_ids)
        for instance in scene.instances:
            if instance.xyz_local is None:
                continue
            samples.append(
                InstancePointCloudSample(
                    scene_id=scene.scene_id,
                    frame_id=scene.frame_id,
                    instance_id=instance.instance_id,
                    semantic_id=instance.semantic_id,
                    xyz_local=instance.xyz_local.astype(np.float32, copy=False),
                    rgb=instance.rgb.astype(np.uint8, copy=False),
                )
            )
    return samples


def collect_node_samples_from_ply_paths(
    ply_paths: list[str],
    *,
    octree_config: OctreeBuildConfig | None = None,
    drop_negative_instance_ids: bool = True,
    include_leaf_only: bool = False,
) -> list[TreeNodePointCloudSample]:
    config = octree_config or OctreeBuildConfig()
    samples: list[TreeNodePointCloudSample] = []
    for path in ply_paths:
        frame = load_ply_frame(path)
        scene = build_scene_frame(frame, drop_negative_instance_ids=drop_negative_instance_ids)
        for instance in scene.instances:
            if instance.xyz_local is None:
                continue
            nodes = build_instance_octree(instance, config)
            for node in nodes:
                if include_leaf_only and node.structure_state != "leaf":
                    continue
                xyz_local = instance.xyz_local[node.point_indices]
                rgb = instance.rgb[node.point_indices]
                samples.append(
                    TreeNodePointCloudSample(
                        scene_id=scene.scene_id,
                        frame_id=scene.frame_id,
                        instance_id=instance.instance_id,
                        semantic_id=instance.semantic_id,
                        node_id=node.node_id,
                        parent_id=node.parent_id,
                        child_index=node.child_index,
                        path_code=node.path_code,
                        level=node.level,
                        split_flag=node.split_flag,
                        center_local=node.center_local.astype(np.float32, copy=False),
                        size_local=node.size_local.astype(np.float32, copy=False),
                        xyz_local=xyz_local.astype(np.float32, copy=False),
                        rgb=rgb.astype(np.uint8, copy=False),
                    )
                )
    return samples


def build_instance_dataset_from_ply_dir(
    ply_dir: str,
    *,
    points_per_instance: int = 256,
    seed: int = 0,
) -> InstancePointCloudDataset:
    paths = sorted(str(path) for path in Path(ply_dir).glob("*.ply"))
    samples = collect_instance_samples_from_ply_paths(paths)
    return InstancePointCloudDataset(samples, points_per_instance=points_per_instance, seed=seed)


def build_instance_dataset_from_ply_paths(
    ply_paths: list[str],
    *,
    points_per_instance: int = 256,
    seed: int = 0,
) -> InstancePointCloudDataset:
    samples = collect_instance_samples_from_ply_paths(ply_paths)
    return InstancePointCloudDataset(samples, points_per_instance=points_per_instance, seed=seed)


def build_node_dataset_from_ply_dir(
    ply_dir: str,
    *,
    points_per_node: int = 128,
    queries_per_node: int = 64,
    udf_truncation: float = 0.25,
    near_surface_threshold: float | None = None,
    seed: int = 0,
    octree_config: OctreeBuildConfig | None = None,
    include_leaf_only: bool = False,
) -> TreeNodePointCloudDataset:
    paths = sorted(str(path) for path in Path(ply_dir).glob("*.ply"))
    samples = collect_node_samples_from_ply_paths(
        paths,
        octree_config=octree_config,
        include_leaf_only=include_leaf_only,
    )
    return TreeNodePointCloudDataset(
        samples,
        points_per_node=points_per_node,
        queries_per_node=queries_per_node,
        udf_truncation=udf_truncation,
        near_surface_threshold=near_surface_threshold,
        seed=seed,
    )


def build_node_dataset_from_ply_paths(
    ply_paths: list[str],
    *,
    points_per_node: int = 128,
    queries_per_node: int = 64,
    udf_truncation: float = 0.25,
    near_surface_threshold: float | None = None,
    seed: int = 0,
    octree_config: OctreeBuildConfig | None = None,
    include_leaf_only: bool = False,
) -> TreeNodePointCloudDataset:
    samples = collect_node_samples_from_ply_paths(
        ply_paths,
        octree_config=octree_config,
        include_leaf_only=include_leaf_only,
    )
    return TreeNodePointCloudDataset(
        samples,
        points_per_node=points_per_node,
        queries_per_node=queries_per_node,
        udf_truncation=udf_truncation,
        near_surface_threshold=near_surface_threshold,
        seed=seed,
    )


def sample_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    num_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] == 0:
        raise ValueError("Cannot sample an empty instance point cloud.")

    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    xyz_sorted = xyz[order]
    rgb_sorted = rgb[order]
    rng = np.random.default_rng(seed)

    if xyz_sorted.shape[0] >= num_points:
        choice = rng.choice(xyz_sorted.shape[0], size=num_points, replace=False)
    else:
        extra = rng.choice(xyz_sorted.shape[0], size=num_points - xyz_sorted.shape[0], replace=True)
        choice = np.concatenate([np.arange(xyz_sorted.shape[0]), extra], axis=0)

    sampled_xyz = xyz_sorted[choice]
    sampled_rgb = rgb_sorted[choice]
    reorder = np.lexsort((sampled_xyz[:, 2], sampled_xyz[:, 1], sampled_xyz[:, 0]))
    return sampled_xyz[reorder], sampled_rgb[reorder]


def build_udf_queries(
    xyz: np.ndarray,
    *,
    num_queries: int,
    truncation_distance: float,
    seed: int,
    expand_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] == 0:
        raise ValueError("Cannot build UDF queries from an empty node point cloud.")

    rng = np.random.default_rng(seed)
    xyz = xyz.astype(np.float32, copy=False)

    bbox_min = np.min(xyz, axis=0)
    bbox_max = np.max(xyz, axis=0)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-2)
    expanded_min = bbox_min - bbox_size * float(expand_ratio)
    expanded_max = bbox_max + bbox_size * float(expand_ratio)

    surface_count = max(1, num_queries // 2)
    volume_count = max(0, num_queries - surface_count)

    surface_choice = rng.choice(xyz.shape[0], size=surface_count, replace=xyz.shape[0] < surface_count)
    surface_queries = xyz[surface_choice] + rng.normal(
        loc=0.0,
        scale=np.maximum(bbox_size * 0.02, 1e-3),
        size=(surface_count, 3),
    ).astype(np.float32)
    volume_queries = rng.uniform(expanded_min, expanded_max, size=(volume_count, 3)).astype(np.float32)

    queries = np.concatenate([surface_queries, volume_queries], axis=0)
    diff = queries[:, None, :] - xyz[None, :, :]
    distances = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1, keepdims=True)).astype(np.float32)
    udf = np.minimum(distances, float(truncation_distance)).astype(np.float32)

    order = rng.permutation(queries.shape[0])
    return queries[order][:num_queries], udf[order][:num_queries]
