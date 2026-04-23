from __future__ import annotations

import numpy as np

from threedvae.data.schema import PlyFrameData, SceneFrame, SceneInstance
from threedvae.geometry.bbox import estimate_ground_aligned_obb
from threedvae.geometry.frames import pose_from_obb, transform_points_to_local


def build_scene_instances(
    frame: PlyFrameData,
    *,
    drop_negative_instance_ids: bool = True,
    semantic_name_lookup: dict[int, str] | None = None,
) -> list[SceneInstance]:
    instance_ids = np.unique(frame.instance_id)
    if drop_negative_instance_ids:
        instance_ids = instance_ids[instance_ids >= 0]

    instances: list[SceneInstance] = []
    for instance_id in instance_ids.tolist():
        mask = frame.instance_id == instance_id
        xyz = frame.xyz[mask]
        rgb = frame.rgb[mask]
        semantic_values = frame.semantic_id[mask]
        semantic_id = int(_majority_vote(semantic_values))
        bbox = estimate_ground_aligned_obb(xyz)
        pose = pose_from_obb(bbox)
        xyz_local = transform_points_to_local(xyz, pose)
        category_name = (
            semantic_name_lookup[semantic_id]
            if semantic_name_lookup and semantic_id in semantic_name_lookup
            else f"semantic_{semantic_id}"
        )

        instances.append(
            SceneInstance(
                instance_id=int(instance_id),
                semantic_id=semantic_id,
                category_name=category_name,
                xyz_ego=xyz.astype(np.float32, copy=False),
                rgb=rgb.astype(np.uint8, copy=False),
                bbox_ego=bbox,
                pose_ego=pose,
                xyz_local=xyz_local,
                num_points=int(xyz.shape[0]),
            )
        )

    instances.sort(key=lambda item: item.instance_id)
    return instances


def build_scene_frame(
    frame: PlyFrameData,
    *,
    drop_negative_instance_ids: bool = True,
    semantic_name_lookup: dict[int, str] | None = None,
) -> SceneFrame:
    instances = build_scene_instances(
        frame,
        drop_negative_instance_ids=drop_negative_instance_ids,
        semantic_name_lookup=semantic_name_lookup,
    )
    return SceneFrame(
        scene_id=frame.scene_id,
        frame_id=frame.frame_id,
        ply_path=frame.ply_path,
        instances=instances,
        num_points=int(frame.xyz.shape[0]),
        num_instances=len(instances),
    )


def _majority_vote(values: np.ndarray) -> int:
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[int(np.argmax(counts))])
