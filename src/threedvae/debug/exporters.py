from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from threedvae.data.schema import InstanceBBoxRecord, OctreeNodeDebugRecord, SceneFrame, SceneInstance
from threedvae.geometry.bbox import obb_vertices, quad_faces


def export_scene_instance_bboxes_ply(scene: SceneFrame, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_instance_ids: list[np.ndarray] = []
    all_semantics: list[np.ndarray] = []
    face_template = quad_faces()

    vertex_offset = 0
    for instance in scene.instances:
        vertices = obb_vertices(instance.bbox_ego)
        color = _color_from_id(instance.semantic_id)
        colors = np.tile(color[None, :], (vertices.shape[0], 1))
        instance_ids = np.full((vertices.shape[0],), instance.instance_id, dtype=np.int32)
        semantics = np.full((vertices.shape[0],), instance.semantic_id, dtype=np.int32)
        faces = face_template + vertex_offset

        all_vertices.append(vertices)
        all_faces.append(faces)
        all_colors.append(colors)
        all_instance_ids.append(instance_ids)
        all_semantics.append(semantics)
        vertex_offset += vertices.shape[0]

    vertices = np.concatenate(all_vertices, axis=0) if all_vertices else np.zeros((0, 3), dtype=np.float32)
    faces = np.concatenate(all_faces, axis=0) if all_faces else np.zeros((0, 4), dtype=np.int32)
    colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.uint8)
    instance_ids = np.concatenate(all_instance_ids, axis=0) if all_instance_ids else np.zeros((0,), dtype=np.int32)
    semantics = np.concatenate(all_semantics, axis=0) if all_semantics else np.zeros((0,), dtype=np.int32)

    _write_ascii_mesh_ply(path, vertices, faces, colors, instance_ids, semantics)


def export_instance_bbox_json(instance: SceneInstance, out_path: str) -> None:
    record = InstanceBBoxRecord(
        instance_id=instance.instance_id,
        semantic_id=instance.semantic_id,
        center_xyz=instance.bbox_ego.center_xyz.copy(),
        size_xyz=instance.bbox_ego.size_xyz.copy(),
        yaw=float(instance.bbox_ego.yaw),
        vertices_ego=obb_vertices(instance.bbox_ego),
        quad_faces=quad_faces(),
    )
    _write_json(Path(out_path), _to_jsonable(record))


def export_instance_octree_debug(
    instance: SceneInstance,
    nodes: list[OctreeNodeDebugRecord],
    out_dir: str,
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"instance_{instance.instance_id}_octree_nodes.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for node in nodes:
            handle.write(json.dumps(_to_jsonable(node), ensure_ascii=False) + "\n")


def _write_ascii_mesh_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    instance_ids: np.ndarray,
    semantics: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property int instance\n")
        handle.write("property int semantic\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")

        for vertex, color, instance_id, semantic_id in zip(vertices, colors, instance_ids, semantics):
            handle.write(
                f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(instance_id)} {int(semantic_id)}\n"
            )

        for face in faces:
            handle.write(f"4 {int(face[0])} {int(face[1])} {int(face[2])} {int(face[3])}\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _to_jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _color_from_id(idx: int) -> np.ndarray:
    # Deterministic but simple semantic color palette for mesh debugging.
    return np.asarray(
        [
            (idx * 53 + 91) % 255,
            (idx * 97 + 17) % 255,
            (idx * 193 + 37) % 255,
        ],
        dtype=np.uint8,
    )

