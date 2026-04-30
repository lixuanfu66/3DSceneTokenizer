from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def split_flag_from_name(name: str) -> int:
    normalized = name.strip().upper()
    mapping = {
        "X": 0b001,
        "Y": 0b010,
        "Z": 0b100,
        "XY": 0b011,
        "XZ": 0b101,
        "YZ": 0b110,
        "XYZ": 0b111,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported split flag name: {name}")
    return mapping[normalized]


@dataclass(slots=True)
class SemanticOctreePolicy:
    tag_name: str
    min_depth: int
    max_depth_by_distance: dict[str, int]
    preferred_split_flag: int = 0b111
    lock_preferred_split: bool = False
    priority: str = "low"
    geom_threshold_scale: float = 1.0
    rgb_threshold_scale: float = 1.0

    def max_depth_for_distance(self, distance_bin: str) -> int:
        return int(self.max_depth_by_distance.get(distance_bin, self.max_depth_by_distance.get("far", 1)))


def build_default_carla_semantic_policies() -> dict[int, SemanticOctreePolicy]:
    return {
        0: SemanticOctreePolicy("Unlabeled", min_depth=0, max_depth_by_distance={"near": 0, "mid": 0, "far": 0}, preferred_split_flag=0b111, lock_preferred_split=True, priority="low"),
        1: SemanticOctreePolicy("Roads", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b011, lock_preferred_split=True, priority="high"),
        2: SemanticOctreePolicy("SideWalks", min_depth=2, max_depth_by_distance={"near": 4, "mid": 3, "far": 2}, preferred_split_flag=0b011, lock_preferred_split=True, priority="high"),
        3: SemanticOctreePolicy("Building", min_depth=0, max_depth_by_distance={"near": 3, "mid": 2, "far": 1}, preferred_split_flag=0b111, lock_preferred_split=False, priority="low"),
        4: SemanticOctreePolicy("Wall", min_depth=0, max_depth_by_distance={"near": 3, "mid": 2, "far": 1}, preferred_split_flag=0b111, lock_preferred_split=False, priority="low"),
        5: SemanticOctreePolicy("Fence", min_depth=0, max_depth_by_distance={"near": 3, "mid": 2, "far": 1}, preferred_split_flag=0b111, lock_preferred_split=False, priority="low"),
        6: SemanticOctreePolicy("Pole", min_depth=0, max_depth_by_distance={"near": 3, "mid": 2, "far": 1}, preferred_split_flag=0b111, lock_preferred_split=False, priority="medium"),
        7: SemanticOctreePolicy("TrafficLight", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        8: SemanticOctreePolicy("TrafficSign", min_depth=1, max_depth_by_distance={"near": 4, "mid": 3, "far": 2}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        9: SemanticOctreePolicy("Vegetation", min_depth=0, max_depth_by_distance={"near": 3, "mid": 2, "far": 1}, preferred_split_flag=0b111, lock_preferred_split=False, priority="low"),
        10: SemanticOctreePolicy("Terrain", min_depth=1, max_depth_by_distance={"near": 3, "mid": 2, "far": 1}, preferred_split_flag=0b011, lock_preferred_split=False, priority="medium"),
        11: SemanticOctreePolicy("Sky", min_depth=0, max_depth_by_distance={"near": 0, "mid": 0, "far": 0}, preferred_split_flag=0b111, lock_preferred_split=True, priority="low"),
        12: SemanticOctreePolicy("Pedestrian", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        13: SemanticOctreePolicy("Rider", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        14: SemanticOctreePolicy("Car", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        15: SemanticOctreePolicy("Truck", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        16: SemanticOctreePolicy("Bus", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        17: SemanticOctreePolicy("Train", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        18: SemanticOctreePolicy("Motorcycle", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        19: SemanticOctreePolicy("Bicycle", min_depth=2, max_depth_by_distance={"near": 5, "mid": 4, "far": 3}, preferred_split_flag=0b111, lock_preferred_split=False, priority="high"),
        20: SemanticOctreePolicy("Static", min_depth=0, max_depth_by_distance={"near": 2, "mid": 1, "far": 1}, preferred_split_flag=0b111, lock_preferred_split=False, priority="low"),
        21: SemanticOctreePolicy("Dynamic", min_depth=1, max_depth_by_distance={"near": 4, "mid": 3, "far": 2}, preferred_split_flag=0b111, lock_preferred_split=False, priority="medium"),
        22: SemanticOctreePolicy("Other", min_depth=0, max_depth_by_distance={"near": 2, "mid": 1, "far": 1}, preferred_split_flag=0b111, lock_preferred_split=False, priority="low"),
        23: SemanticOctreePolicy("Water", min_depth=0, max_depth_by_distance={"near": 1, "mid": 1, "far": 0}, preferred_split_flag=0b011, lock_preferred_split=True, priority="low"),
        24: SemanticOctreePolicy("RoadLine", min_depth=3, max_depth_by_distance={"near": 6, "mid": 5, "far": 4}, preferred_split_flag=0b011, lock_preferred_split=True, priority="high"),
        25: SemanticOctreePolicy("Ground", min_depth=1, max_depth_by_distance={"near": 3, "mid": 2, "far": 1}, preferred_split_flag=0b011, lock_preferred_split=True, priority="medium"),
        26: SemanticOctreePolicy("Bridge", min_depth=0, max_depth_by_distance={"near": 2, "mid": 2, "far": 1}, preferred_split_flag=0b111, lock_preferred_split=False, priority="low"),
        27: SemanticOctreePolicy("RailTrack", min_depth=1, max_depth_by_distance={"near": 3, "mid": 2, "far": 1}, preferred_split_flag=0b011, lock_preferred_split=False, priority="medium"),
        28: SemanticOctreePolicy("GuardRail", min_depth=1, max_depth_by_distance={"near": 4, "mid": 3, "far": 2}, preferred_split_flag=0b111, lock_preferred_split=False, priority="medium"),
    }


def build_default_object_semantic_policies(
    *,
    object_semantic_id: int = 100,
) -> dict[int, SemanticOctreePolicy]:
    return {
        int(object_semantic_id): SemanticOctreePolicy(
            "Object",
            min_depth=2,
            max_depth_by_distance={"near": 4, "mid": 4, "far": 4},
            preferred_split_flag=0b111,
            lock_preferred_split=False,
            priority="high",
            geom_threshold_scale=0.8,
            rgb_threshold_scale=1.0,
        )
    }


@dataclass(slots=True)
class OctreeBuildConfig:
    near_distance: float = 15.0
    mid_distance: float = 35.0
    min_points_per_node: int = 8
    min_points_per_leaf: int = 1
    geom_threshold: float = 0.35
    rgb_threshold: float = 0.10
    extent_weight: float = 0.45
    occupancy_weight: float = 0.35
    plane_residual_weight: float = 0.20
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
    semantic_policies: dict[int, SemanticOctreePolicy] = field(default_factory=dict)
    high_priority_semantics: set[int] = field(default_factory=set)
    medium_priority_semantics: set[int] = field(default_factory=set)
    xy_only_semantics: set[int] = field(default_factory=set)
    xz_only_semantics: set[int] = field(default_factory=set)
    yz_only_semantics: set[int] = field(default_factory=set)
    planar_axis_ratio_threshold: float = 0.15

    @classmethod
    def with_default_carla_semantics(cls, **kwargs) -> "OctreeBuildConfig":
        return cls(semantic_policies=build_default_carla_semantic_policies(), **kwargs)

    @classmethod
    def with_default_object_semantics(cls, **kwargs) -> "OctreeBuildConfig":
        kwargs.setdefault("min_points_per_node", 10)
        kwargs.setdefault("min_points_per_leaf", 4)
        return cls(semantic_policies=build_default_object_semantic_policies(), **kwargs)

    def distance_bin(self, center_xyz: np.ndarray) -> str:
        distance = float(np.linalg.norm(center_xyz[:2]))
        if distance < self.near_distance:
            return "near"
        if distance < self.mid_distance:
            return "mid"
        return "far"

    def policy_for(self, semantic_id: int) -> SemanticOctreePolicy | None:
        return self.semantic_policies.get(int(semantic_id))

    def semantic_priority(self, semantic_id: int) -> str:
        policy = self.policy_for(semantic_id)
        if policy is not None:
            return policy.priority
        if semantic_id in self.high_priority_semantics:
            return "high"
        if semantic_id in self.medium_priority_semantics:
            return "medium"
        return "low"

    def max_depth_for(self, semantic_id: int, center_xyz: np.ndarray) -> int:
        distance_bin = self.distance_bin(center_xyz)
        policy = self.policy_for(semantic_id)
        if policy is not None:
            return policy.max_depth_for_distance(distance_bin)
        priority = self.semantic_priority(semantic_id)
        return int(self.max_depth_by_priority_distance[priority][distance_bin])

    def min_depth_for(self, semantic_id: int) -> int:
        policy = self.policy_for(semantic_id)
        if policy is not None:
            return int(policy.min_depth)
        return int(self.min_depth_by_priority[self.semantic_priority(semantic_id)])

    def split_thresholds_for(self, semantic_id: int) -> tuple[float, float]:
        policy = self.policy_for(semantic_id)
        if policy is None:
            return float(self.geom_threshold), float(self.rgb_threshold)
        return (
            float(self.geom_threshold * policy.geom_threshold_scale),
            float(self.rgb_threshold * policy.rgb_threshold_scale),
        )

    def split_flag_for(
        self,
        semantic_id: int,
        occupied_extent_xyz: np.ndarray,
        axis_std_xyz: np.ndarray | None = None,
    ) -> int:
        policy = self.policy_for(semantic_id)
        if policy is not None and policy.lock_preferred_split:
            return int(policy.preferred_split_flag)

        if semantic_id in self.xy_only_semantics:
            return 0b011
        if semantic_id in self.xz_only_semantics:
            return 0b101
        if semantic_id in self.yz_only_semantics:
            return 0b110

        geometric_flag = self._geometry_split_flag(occupied_extent_xyz, axis_std_xyz)
        if policy is None:
            return geometric_flag
        return self._merge_preferred_and_geometric(policy.preferred_split_flag, geometric_flag)

    def _geometry_split_flag(
        self,
        occupied_extent_xyz: np.ndarray,
        axis_std_xyz: np.ndarray | None = None,
    ) -> int:
        extent = np.maximum(occupied_extent_xyz.astype(np.float64, copy=False), 1e-6)
        max_extent = float(np.max(extent))
        if max_extent <= 0.0:
            return 0b111

        extent_ratios = extent / max_extent
        axis_signal = extent_ratios
        if axis_std_xyz is not None and axis_std_xyz.shape[0] == 3:
            axis_std = np.maximum(axis_std_xyz.astype(np.float64, copy=False), 1e-6)
            max_std = float(np.max(axis_std))
            if max_std > 0.0:
                axis_signal = np.maximum(axis_signal, axis_std / max_std)

        weakest_axis = int(np.argmin(axis_signal))
        weakest_ratio = float(axis_signal[weakest_axis])
        if weakest_ratio < self.planar_axis_ratio_threshold:
            active_axes = [axis for axis in range(3) if axis != weakest_axis]
        else:
            active_axes = [0, 1, 2]

        split_flag = 0
        for axis in active_axes:
            split_flag |= 1 << axis
        return int(split_flag)

    @staticmethod
    def _merge_preferred_and_geometric(preferred_flag: int, geometric_flag: int) -> int:
        preferred_flag = int(preferred_flag)
        geometric_flag = int(geometric_flag)
        if preferred_flag == 0b111:
            return geometric_flag
        if geometric_flag == 0b111:
            return 0b111
        return preferred_flag
