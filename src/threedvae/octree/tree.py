from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from threedvae.data.schema import OctreeNodeDebugRecord, SceneInstance
from threedvae.geometry.bbox import quad_faces
from threedvae.octree.split_policy import OctreeBuildConfig


@dataclass(slots=True)
class OctreeNode:
    node_id: int
    level: int
    morton_code: int
    parent_id: int | None
    child_index: int | None
    path_code: str
    split_flag: int
    child_mask: int
    visibility_state: str
    center_local: np.ndarray
    size_local: np.ndarray
    point_indices: np.ndarray
    geom_score: float
    rgb_score: float
    structure_state: str


@dataclass(slots=True)
class NodeComplexity:
    split_flag: int
    extent_score: float
    occupancy_score: float
    plane_residual_score: float
    geom_score: float
    rgb_score: float


def build_instance_octree(
    instance: SceneInstance,
    config: OctreeBuildConfig | None = None,
) -> list[OctreeNode]:
    if instance.xyz_local is None:
        raise ValueError("SceneInstance.xyz_local must be available before building an octree.")

    config = config or OctreeBuildConfig()
    semantic_id = int(instance.semantic_id)
    max_depth = config.max_depth_for(semantic_id, instance.pose_ego.center_xyz)
    min_depth = config.min_depth_for(semantic_id)

    root_center = np.zeros(3, dtype=np.float32)
    root_size = np.maximum(instance.bbox_ego.size_xyz.astype(np.float32, copy=True), 1e-3)
    all_indices = np.arange(instance.xyz_local.shape[0], dtype=np.int32)

    nodes: list[OctreeNode] = []
    next_id = 0

    def recurse(
        center_local: np.ndarray,
        size_local: np.ndarray,
        point_indices: np.ndarray,
        *,
        level: int,
        morton_code: int,
        parent_id: int | None,
        child_index: int | None,
        path_code: str,
    ) -> None:
        nonlocal next_id

        xyz_points = instance.xyz_local[point_indices]
        complexity = compute_node_complexity(
            xyz_points=xyz_points,
            rgb=instance.rgb[point_indices],
            center_local=center_local,
            size_local=size_local,
            semantic_id=semantic_id,
            config=config,
        )
        structure_state = _decide_split(
            level=level,
            min_depth=min_depth,
            max_depth=max_depth,
            point_count=int(point_indices.shape[0]),
            semantic_id=semantic_id,
            geom_score=complexity.geom_score,
            rgb_score=complexity.rgb_score,
            split_flag=complexity.split_flag,
            config=config,
        )

        node_id = next_id
        next_id += 1
        node = OctreeNode(
            node_id=node_id,
            level=level,
            morton_code=morton_code,
            parent_id=parent_id,
            child_index=child_index,
            path_code=path_code,
            split_flag=complexity.split_flag,
            child_mask=0,
            visibility_state=_visibility_state_from_points(int(point_indices.shape[0])),
            center_local=center_local.astype(np.float32, copy=False),
            size_local=size_local.astype(np.float32, copy=False),
            point_indices=point_indices.astype(np.int32, copy=False),
            geom_score=float(complexity.geom_score),
            rgb_score=float(complexity.rgb_score),
            structure_state=structure_state,
        )
        nodes.append(node)

        if structure_state != "split":
            return

        child_size = _child_size(size_local, complexity.split_flag)
        xyz = xyz_points

        child_masks = _child_masks(xyz, center_local, complexity.split_flag)
        materialized_child_mask = 0
        child_entries: list[tuple[int, np.ndarray]] = []
        for child_index in _child_indices(complexity.split_flag):
            masked_indices = child_masks[child_index]
            child_indices = point_indices[masked_indices]
            if child_indices.size == 0:
                continue
            materialized_child_mask |= 1 << _local_child_slot(complexity.split_flag, child_index)
            child_entries.append((child_index, child_indices))

        node.child_mask = int(materialized_child_mask)
        for child_index, child_indices in child_entries:
            offset = _child_offset(size_local, child_index, complexity.split_flag)
            child_path = f"{path_code}.{child_index}" if path_code else str(child_index)
            recurse(
                center_local=center_local + offset,
                size_local=child_size,
                point_indices=child_indices,
                level=level + 1,
                morton_code=(morton_code << 3) | child_index,
                parent_id=node_id,
                child_index=child_index,
                path_code=child_path,
            )

    recurse(
        center_local=root_center,
        size_local=root_size,
        point_indices=all_indices,
        level=0,
        morton_code=0,
        parent_id=None,
        child_index=None,
        path_code="root",
    )
    return nodes


def build_octree_debug_records(instance: SceneInstance, nodes: list[OctreeNode]) -> list[OctreeNodeDebugRecord]:
    records: list[OctreeNodeDebugRecord] = []
    faces = quad_faces()

    for node in nodes:
        vertices = cuboid_vertices(node.center_local, node.size_local)
        records.append(
            OctreeNodeDebugRecord(
                instance_id=instance.instance_id,
                node_id=node.node_id,
                parent_id=node.parent_id,
                child_index=node.child_index,
                path_code=node.path_code,
                level=node.level,
                morton_code=node.morton_code,
                split_flag=node.split_flag,
                child_mask=node.child_mask,
                visibility_state=node.visibility_state,
                semantic_id=instance.semantic_id,
                bbox_center_local=node.center_local.copy(),
                bbox_size_local=node.size_local.copy(),
                bbox_corners_local=vertices.copy(),
                vertices_local=vertices.copy(),
                quad_faces=faces.copy(),
                num_points=int(node.point_indices.shape[0]),
                structure_state=node.structure_state,
            )
        )
    return records


def cuboid_vertices(center_local: np.ndarray, size_local: np.ndarray) -> np.ndarray:
    half = 0.5 * size_local.astype(np.float64, copy=False)
    local_vertices = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    return (local_vertices + center_local.astype(np.float64, copy=False)).astype(np.float32)


def geometry_complexity(xyz_local: np.ndarray, size_local: np.ndarray) -> float:
    return geometry_complexity_from_extent(occupied_extent_xyz(xyz_local), size_local)


def geometry_complexity_from_extent(occupied_extent: np.ndarray, size_local: np.ndarray) -> float:
    if occupied_extent.shape[0] == 0:
        return 0.0
    occupied_extent = np.maximum(occupied_extent.astype(np.float64, copy=False), 1e-6)
    normalized_extent = occupied_extent / np.maximum(size_local.astype(np.float64, copy=False), 1e-6)
    return float(np.clip(normalized_extent.max(), 0.0, 1.0))


def occupied_extent_xyz(xyz_local: np.ndarray) -> np.ndarray:
    if xyz_local.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.maximum(xyz_local.max(axis=0) - xyz_local.min(axis=0), 1e-6).astype(np.float32)


def rgb_variation(rgb: np.ndarray) -> float:
    if rgb.shape[0] == 0:
        return 0.0
    return float(np.mean(np.std(rgb.astype(np.float32) / 255.0, axis=0)))


def compute_node_complexity(
    *,
    xyz_points: np.ndarray,
    rgb: np.ndarray,
    center_local: np.ndarray,
    size_local: np.ndarray,
    semantic_id: int,
    config: OctreeBuildConfig,
) -> NodeComplexity:
    occupied_extent = occupied_extent_xyz(xyz_points)
    axis_std = axis_std_xyz(xyz_points)
    split_flag = config.split_flag_for(semantic_id, occupied_extent, axis_std)
    extent_score = extent_complexity(occupied_extent, size_local, split_flag)
    occupancy_score = occupancy_complexity(xyz_points, center_local, split_flag)
    plane_residual_score = plane_residual_complexity(xyz_points, size_local)
    geom_score = float(
        np.clip(
            config.extent_weight * extent_score
            + config.occupancy_weight * occupancy_score
            + config.plane_residual_weight * plane_residual_score,
            0.0,
            1.0,
        )
    )
    return NodeComplexity(
        split_flag=split_flag,
        extent_score=float(extent_score),
        occupancy_score=float(occupancy_score),
        plane_residual_score=float(plane_residual_score),
        geom_score=geom_score,
        rgb_score=rgb_variation(rgb),
    )


def axis_std_xyz(xyz_local: np.ndarray) -> np.ndarray:
    if xyz_local.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32)
    return np.std(xyz_local.astype(np.float32, copy=False), axis=0).astype(np.float32)


def extent_complexity(occupied_extent: np.ndarray, size_local: np.ndarray, split_flag: int) -> float:
    if occupied_extent.shape[0] == 0 or split_flag == 0:
        return 0.0

    active_axes = [axis for axis in range(3) if split_flag & (1 << axis)]
    if not active_axes:
        return 0.0

    occupied = np.maximum(occupied_extent.astype(np.float64, copy=False), 1e-6)
    size = np.maximum(size_local.astype(np.float64, copy=False), 1e-6)
    normalized_extent = occupied[active_axes] / size[active_axes]
    return float(np.clip(np.mean(normalized_extent), 0.0, 1.0))


def occupancy_complexity(xyz_local: np.ndarray, center_local: np.ndarray, split_flag: int) -> float:
    if xyz_local.shape[0] == 0 or split_flag == 0:
        return 0.0

    child_masks = _child_masks(xyz_local, center_local, split_flag)
    active_slots = 1 << len([axis for axis in range(3) if split_flag & (1 << axis)])
    occupied_slots = 0
    for child_index in _child_indices(split_flag):
        if np.any(child_masks[child_index]):
            occupied_slots += 1
    if active_slots <= 0:
        return 0.0
    return float(np.clip(occupied_slots / active_slots, 0.0, 1.0))


def plane_residual_complexity(xyz_local: np.ndarray, size_local: np.ndarray) -> float:
    if xyz_local.shape[0] < 3:
        return 0.0

    centered = xyz_local.astype(np.float64, copy=False) - np.mean(xyz_local.astype(np.float64, copy=False), axis=0, keepdims=True)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0

    if vh.shape[0] < 3:
        return 0.0
    normal = vh[-1]
    distances = centered @ normal
    rmse = float(np.sqrt(np.mean(np.square(distances))))
    normalizer = max(float(np.linalg.norm(size_local.astype(np.float64, copy=False))), 1e-6)
    return float(np.clip(rmse / normalizer, 0.0, 1.0))


def _decide_split(
    *,
    level: int,
    min_depth: int,
    max_depth: int,
    point_count: int,
    semantic_id: int,
    geom_score: float,
    rgb_score: float,
    split_flag: int,
    config: OctreeBuildConfig,
) -> str:
    geom_threshold, rgb_threshold = config.split_thresholds_for(semantic_id)
    if point_count == 0:
        return "leaf"
    if split_flag == 0:
        return "leaf"
    if level >= max_depth:
        return "leaf"
    if point_count < config.min_points_per_node:
        return "leaf"
    if level < min_depth:
        return "split"
    if geom_score > geom_threshold:
        return "split"
    if rgb_score > rgb_threshold:
        return "split"
    return "leaf"


def _child_indices(split_flag: int) -> list[int]:
    active_axes = [axis for axis in range(3) if split_flag & (1 << axis)]
    child_indices: list[int] = []
    for local_index in range(1 << len(active_axes)):
        child_index = 0
        for bit_position, axis in enumerate(active_axes):
            if local_index & (1 << bit_position):
                child_index |= 1 << axis
        child_indices.append(child_index)
    return child_indices


def _local_child_slot(split_flag: int, child_index: int) -> int:
    active_axes = [axis for axis in range(3) if split_flag & (1 << axis)]
    local_slot = 0
    for bit_position, axis in enumerate(active_axes):
        if child_index & (1 << axis):
            local_slot |= 1 << bit_position
    return local_slot


def _child_masks(xyz_local: np.ndarray, center_local: np.ndarray, split_flag: int) -> list[np.ndarray]:
    x_high = xyz_local[:, 0] >= center_local[0]
    y_high = xyz_local[:, 1] >= center_local[1]
    z_high = xyz_local[:, 2] >= center_local[2]
    masks: list[np.ndarray] = []
    for child_index in range(8):
        mask = np.ones((xyz_local.shape[0],), dtype=bool)
        if split_flag & 0b001:
            mask &= (x_high if (child_index & 1) else (~x_high))
        if split_flag & 0b010:
            mask &= (y_high if (child_index & 2) else (~y_high))
        if split_flag & 0b100:
            mask &= (z_high if (child_index & 4) else (~z_high))
        masks.append(mask)
    return masks


def _child_size(size_local: np.ndarray, split_flag: int) -> np.ndarray:
    child_size = size_local.astype(np.float32, copy=True)
    if split_flag & 0b001:
        child_size[0] *= 0.5
    if split_flag & 0b010:
        child_size[1] *= 0.5
    if split_flag & 0b100:
        child_size[2] *= 0.5
    return child_size


def _child_offset(size_local: np.ndarray, child_index: int, split_flag: int) -> np.ndarray:
    offset = np.zeros((3,), dtype=np.float32)
    if split_flag & 0b001:
        offset[0] = -0.25 * size_local[0] if (child_index & 1) == 0 else 0.25 * size_local[0]
    if split_flag & 0b010:
        offset[1] = -0.25 * size_local[1] if (child_index & 2) == 0 else 0.25 * size_local[1]
    if split_flag & 0b100:
        offset[2] = -0.25 * size_local[2] if (child_index & 4) == 0 else 0.25 * size_local[2]
    return offset


def _visibility_state_from_points(point_count: int) -> str:
    if point_count > 0:
        return "observed_surface"
    return "unknown"
