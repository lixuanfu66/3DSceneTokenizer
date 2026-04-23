from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PlyFrameData:
    scene_id: str
    frame_id: str
    ply_path: str
    xyz: np.ndarray
    rgb: np.ndarray
    instance_id: np.ndarray
    semantic_id: np.ndarray


@dataclass(slots=True)
class Pose3DYaw:
    center_xyz: np.ndarray
    yaw: float


@dataclass(slots=True)
class GroundAlignedOBB:
    center_xyz: np.ndarray
    size_xyz: np.ndarray
    yaw: float


@dataclass(slots=True)
class LocalFrameTransform:
    translation_xyz: np.ndarray
    yaw: float


@dataclass(slots=True)
class SceneInstance:
    instance_id: int
    semantic_id: int
    category_name: str
    xyz_ego: np.ndarray
    rgb: np.ndarray
    bbox_ego: GroundAlignedOBB
    pose_ego: Pose3DYaw
    xyz_local: np.ndarray | None = None
    num_points: int = 0
    is_dynamic: bool = False
    confidence: float | None = None


@dataclass(slots=True)
class SceneFrame:
    scene_id: str
    frame_id: str
    ply_path: str
    instances: list[SceneInstance]
    num_points: int
    num_instances: int


@dataclass(slots=True)
class InstanceBBoxRecord:
    instance_id: int
    semantic_id: int
    center_xyz: np.ndarray
    size_xyz: np.ndarray
    yaw: float
    vertices_ego: np.ndarray
    quad_faces: np.ndarray


@dataclass(slots=True)
class OctreeNodeDebugRecord:
    instance_id: int
    node_id: int
    parent_id: int | None
    child_index: int | None
    path_code: str
    level: int
    morton_code: int
    split_flag: int
    child_mask: int
    visibility_state: str
    semantic_id: int
    bbox_center_local: np.ndarray
    bbox_size_local: np.ndarray
    bbox_corners_local: np.ndarray
    vertices_local: np.ndarray
    quad_faces: np.ndarray
    num_points: int
    structure_state: str


@dataclass(slots=True)
class PoseToken:
    instance_id: int
    semantic_id: int
    center_xyz: np.ndarray
    yaw: float
    box_size_xyz: np.ndarray


@dataclass(slots=True)
class InstanceNodeToken:
    node_id: int
    parent_id: int | None
    child_index: int | None
    path_code: str
    level: int
    structure_token: int
    split_flag: int
    child_mask: int
    visibility_state: str
    geom_code: int | None
    rgb_code: int | None
    num_points: int
    learned_code: int | None = None
    code_source: str = "rule"


@dataclass(slots=True)
class InstanceTokenBlock:
    instance_id: int
    semantic_id: int
    structure_type: str
    root_center_local: np.ndarray
    root_size_local: np.ndarray
    max_depth: int
    tokens: list[InstanceNodeToken]


@dataclass(slots=True)
class CompactInstanceNodeToken:
    level: int
    split_flag: int
    child_mask: int
    geom_code: int | None
    rgb_code: int | None
    num_points: int
    learned_code: int | None = None
    code_source: str = "rule"


@dataclass(slots=True)
class CompactInstanceTokenBlock:
    instance_id: int
    semantic_id: int
    structure_type: str
    root_center_local: np.ndarray
    root_size_local: np.ndarray
    max_depth: int
    traversal_order: str
    child_slot_order: str
    tokens: list[CompactInstanceNodeToken]


@dataclass(slots=True)
class SceneTokenPacket:
    scene_id: str
    frame_id: str
    pose_tokens: list[PoseToken]
    instance_blocks: list[InstanceTokenBlock]


@dataclass(slots=True)
class CompactSceneTokenPacket:
    scene_id: str
    frame_id: str
    pose_tokens: list[PoseToken]
    instance_blocks: list[CompactInstanceTokenBlock]


@dataclass(slots=True)
class CompactInstanceHeaderTokenV2:
    semantic_id: int
    center_xyz_q: list[int]
    yaw_q: int
    box_size_xyz_q: list[int]
    node_count: int
    main_code_scheme: str


@dataclass(slots=True)
class CompactInstanceNodeTokenV2:
    split_flag: int
    child_mask: int
    main_code: int


@dataclass(slots=True)
class CompactInstanceTokenBlockV2:
    header: CompactInstanceHeaderTokenV2
    tokens: list[CompactInstanceNodeTokenV2]


@dataclass(slots=True)
class CompactSceneTokenPacketV2:
    scene_id: str
    frame_id: str
    pose_quantization: dict[str, float]
    instance_blocks: list[CompactInstanceTokenBlockV2]


@dataclass(slots=True)
class LinearizedSceneToken:
    token_type: str
    payload: dict[str, object]


@dataclass(slots=True)
class SerializedTokenSequence:
    scene_id: str
    frame_id: str
    sequence_format: str
    tokens: list[LinearizedSceneToken]
