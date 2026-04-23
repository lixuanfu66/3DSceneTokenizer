from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class OctreeBuildConfig:
    near_distance: float = 15.0
    mid_distance: float = 35.0
    min_points_per_node: int = 8
    geom_threshold: float = 0.35
    rgb_threshold: float = 0.10
    min_depth_by_priority: dict[str, int] = field(
        default_factory=lambda: {"high": 2, "medium": 1, "low": 0}
    )
    max_depth_by_priority_distance: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "high": {"near": 4, "mid": 3, "far": 2},
            "medium": {"near": 3, "mid": 2, "far": 2},
            "low": {"near": 2, "mid": 1, "far": 1},
        }
    )
    high_priority_semantics: set[int] = field(default_factory=set)
    medium_priority_semantics: set[int] = field(default_factory=set)
    xy_only_semantics: set[int] = field(default_factory=set)
    xz_only_semantics: set[int] = field(default_factory=set)
    yz_only_semantics: set[int] = field(default_factory=set)
    planar_axis_ratio_threshold: float = 0.15

    def semantic_priority(self, semantic_id: int) -> str:
        if semantic_id in self.high_priority_semantics:
            return "high"
        if semantic_id in self.medium_priority_semantics:
            return "medium"
        return "low"

    def distance_bin(self, center_xyz: np.ndarray) -> str:
        distance = float(np.linalg.norm(center_xyz[:2]))
        if distance < self.near_distance:
            return "near"
        if distance < self.mid_distance:
            return "mid"
        return "far"

    def max_depth_for(self, semantic_id: int, center_xyz: np.ndarray) -> int:
        priority = self.semantic_priority(semantic_id)
        distance_bin = self.distance_bin(center_xyz)
        return int(self.max_depth_by_priority_distance[priority][distance_bin])

    def min_depth_for(self, semantic_id: int) -> int:
        return int(self.min_depth_by_priority[self.semantic_priority(semantic_id)])

    def split_flag_for(self, semantic_id: int, occupied_extent_xyz: np.ndarray) -> int:
        if semantic_id in self.xy_only_semantics:
            return 0b011
        if semantic_id in self.xz_only_semantics:
            return 0b101
        if semantic_id in self.yz_only_semantics:
            return 0b110

        extent = np.maximum(occupied_extent_xyz.astype(np.float64, copy=False), 1e-6)
        max_extent = float(np.max(extent))
        if max_extent <= 0.0:
            return 0b111

        ratios = extent / max_extent
        active_axes = [axis for axis, ratio in enumerate(ratios.tolist()) if ratio >= self.planar_axis_ratio_threshold]
        if not active_axes:
            active_axes = [int(np.argmax(extent))]

        split_flag = 0
        for axis in active_axes:
            split_flag |= 1 << axis
        return int(split_flag)
