from __future__ import annotations

import argparse
import struct
import json
from pathlib import Path
import re
import urllib.request

import DracoPy
import numpy as np
import pandas as pd
import trimesh

from threedvae.data.bench2drive_rgbd import write_point_cloud_ply


SMITHSONIAN_METADATA_URL = "https://huggingface.co/datasets/allenai/objaverse-xl/resolve/main/smithsonian/smithsonian.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a tiny Objaverse-XL Smithsonian smoke-test PLY set.")
    parser.add_argument("--out-dir", default="data/objaverse_xl_smoke")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--points-per-object", type=int, default=50000)
    parser.add_argument("--semantic-id", type=int, default=100)
    parser.add_argument("--normalize-unit-box", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    metadata_path = out_dir / "metadata" / "smithsonian.parquet"
    assets_dir = out_dir / "assets"
    ply_dir = out_dir / "ply"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        print(f"Downloading metadata: {SMITHSONIAN_METADATA_URL}")
        urllib.request.urlretrieve(SMITHSONIAN_METADATA_URL, metadata_path)

    rows = pd.read_parquet(metadata_path).iloc[args.start_index : args.start_index + args.count]
    manifest: list[dict[str, object]] = []
    for local_idx, row in enumerate(rows.itertuples(index=False), start=args.start_index):
        title = _metadata_title(row.metadata)
        asset_path = assets_dir / f"{local_idx:05d}_{_slugify(title)}.{row.fileType}"
        if not asset_path.exists():
            print(f"Downloading {local_idx}: {title}")
            urllib.request.urlretrieve(row.fileIdentifier, asset_path)

        mesh = _load_as_mesh(asset_path)
        xyz, rgb = _sample_mesh(mesh, args.points_per_object)
        if args.normalize_unit_box:
            xyz = _normalize_unit_box(xyz)

        instance_id = np.full((xyz.shape[0],), local_idx + 1, dtype=np.int32)
        semantic_id = np.full((xyz.shape[0],), int(args.semantic_id), dtype=np.int32)
        ply_path = ply_dir / f"{local_idx:05d}_{_slugify(title)}.ply"
        write_point_cloud_ply(ply_path, xyz, rgb, instance_id, semantic_id)

        record = {
            "index": local_idx,
            "title": title,
            "source_url": row.fileIdentifier,
            "asset_path": str(asset_path),
            "ply_path": str(ply_path),
            "points": int(xyz.shape[0]),
            "semantic_id": int(args.semantic_id),
            "instance_id": int(local_idx + 1),
            "license": row.license,
            "sha256": row.sha256,
        }
        manifest.append(record)
        print(f"Wrote {ply_path} ({xyz.shape[0]} points)")

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote manifest: {manifest_path}")


def _metadata_title(raw: object) -> str:
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        title = payload.get("title") if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        title = None
    return str(title or "object")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_")
    return slug[:80] or "object"


def _load_as_mesh(path: Path) -> trimesh.Trimesh:
    if path.suffix.lower() == ".glb":
        draco_mesh = _load_draco_glb(path)
        if draco_mesh is not None:
            return draco_mesh

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        dumped = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
        mesh = dumped if isinstance(dumped, trimesh.Trimesh) else trimesh.util.concatenate(dumped)
    if mesh.faces.size == 0 or mesh.vertices.size == 0:
        raise ValueError(f"Asset did not contain a non-empty mesh: {path}")
    return mesh


def _load_draco_glb(path: Path) -> trimesh.Trimesh | None:
    data = path.read_bytes()
    if len(data) < 20:
        return None
    magic, _version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        return None

    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        return None
    payload = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    if "KHR_draco_mesh_compression" not in payload.get("extensionsUsed", []):
        return None

    bin_offset = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", data, bin_offset)
    if bin_type != 0x004E4942:
        return None
    binary = data[bin_offset + 8 : bin_offset + 8 + bin_len]

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0
    for mesh_payload in payload.get("meshes", []):
        for primitive in mesh_payload.get("primitives", []):
            extension = primitive.get("extensions", {}).get("KHR_draco_mesh_compression")
            if extension is None:
                continue
            view = payload["bufferViews"][extension["bufferView"]]
            start = int(view.get("byteOffset", 0))
            length = int(view["byteLength"])
            decoded = DracoPy.decode(binary[start : start + length])
            vertices = np.asarray(decoded.points, dtype=np.float32)
            faces = np.asarray(decoded.faces, dtype=np.int64)
            if vertices.size == 0 or faces.size == 0:
                continue
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += vertices.shape[0]

    if not all_vertices:
        return None
    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _sample_mesh(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    points, face_index = trimesh.sample.sample_surface(mesh, int(count))
    colors = _sample_colors(mesh, face_index)
    return points.astype(np.float32), colors


def _sample_colors(mesh: trimesh.Trimesh, face_index: np.ndarray) -> np.ndarray:
    visual = mesh.visual
    if hasattr(visual, "to_color"):
        try:
            visual = visual.to_color()
        except Exception:
            pass
    try:
        face_colors = getattr(visual, "face_colors", None)
    except Exception:
        face_colors = None
    if face_colors is not None and len(face_colors) == len(mesh.faces):
        return np.asarray(face_colors[face_index, :3], dtype=np.uint8)
    try:
        vertex_colors = getattr(visual, "vertex_colors", None)
    except Exception:
        vertex_colors = None
    if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
        vertex_idx = mesh.faces[face_index, 0]
        return np.asarray(vertex_colors[vertex_idx, :3], dtype=np.uint8)
    return np.full((face_index.shape[0], 3), 180, dtype=np.uint8)


def _normalize_unit_box(xyz: np.ndarray) -> np.ndarray:
    center = (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
    extent = np.max(xyz.max(axis=0) - xyz.min(axis=0))
    if extent <= 0:
        return xyz.astype(np.float32)
    return ((xyz - center) / extent).astype(np.float32)


if __name__ == "__main__":
    main()
