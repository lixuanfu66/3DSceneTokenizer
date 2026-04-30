from __future__ import annotations

import argparse
import gzip
import json
import re
import struct
import urllib.request
from pathlib import Path
from typing import Iterable

import DracoPy
import numpy as np
import trimesh

from threedvae.data.bench2drive_rgbd import write_point_cloud_ply


OBJAVERSE_PP_ANNOTATION_URL = (
    "https://huggingface.co/datasets/cindyxl/ObjaversePlusPlus/resolve/main/annotated_800k.json"
)
OBJAVERSE_OBJECT_PATHS_URL = (
    "https://huggingface.co/datasets/allenai/objaverse/resolve/main/object-paths.json.gz"
)
OBJAVERSE_LVIS_ANNOTATIONS_URL = (
    "https://huggingface.co/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz"
)
OBJAVERSE_ASSET_BASE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"


DRIVING_LVIS_CATEGORIES = {
    "ambulance",
    "bicycle",
    "bus_(vehicle)",
    "car_(automobile)",
    "dirt_bike",
    "fire_engine",
    "garbage_truck",
    "golfcart",
    "minibike",
    "motor_scooter",
    "motorcycle",
    "person",
    "pickup_truck",
    "police_car",
    "race_car",
    "school_bus",
    "signboard",
    "stop_sign",
    "street_sign",
    "tow_truck",
    "traffic_light",
    "trailer_truck",
    "truck",
    "wheelchair",
}


INDOOR_LVIS_CATEGORIES = {
    "armchair",
    "bed",
    "bookcase",
    "bookshelf",
    "bunk_bed",
    "cabinet",
    "cellular_telephone",
    "chair",
    "coffee_table",
    "computer_keyboard",
    "couch",
    "desk",
    "desk_chair",
    "dining_table",
    "file_cabinet",
    "folding_chair",
    "highchair",
    "kitchen_table",
    "lamp",
    "laptop_computer",
    "monitor_(computer_equipment) computer_monitor",
    "mouse_(computer_equipment)",
    "office_chair",
    "printer",
    "rocking_chair",
    "router_(computer_equipment)",
    "sofa",
    "sofa_bed",
    "stool",
    "table",
    "table_lamp",
    "telephone",
    "tv_monitor",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare high-quality Objaverse assets selected by Objaverse++ annotations."
    )
    parser.add_argument("--out-dir", default="data/objaverse_pp_quality")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--per-category-count",
        type=int,
        default=0,
        help="If set with category filtering, select up to this many candidates per LVIS category.",
    )
    parser.add_argument(
        "--target-success-count",
        type=int,
        default=0,
        help="Stop after this many assets have been successfully converted. Useful when some candidates fail filters.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--points-per-object", type=int, default=50000)
    parser.add_argument("--semantic-id", type=int, default=100)
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--styles", default="realistic,scanned")
    parser.add_argument("--densities", default="mid,high")
    parser.add_argument(
        "--category-preset",
        default="none",
        choices=["none", "driving", "indoor", "driving,indoor", "indoor,driving"],
        help="Optional LVIS category preset to intersect with Objaverse++ quality filtering.",
    )
    parser.add_argument(
        "--categories",
        default="",
        help="Comma-separated LVIS category whitelist. Applied in addition to --category-preset.",
    )
    parser.add_argument(
        "--require-color-variance",
        action="store_true",
        help="Skip sampled objects whose exported RGB is nearly constant.",
    )
    parser.add_argument("--min-color-std", type=float, default=5.0)
    parser.add_argument("--max-asset-mb", type=float, default=128.0)
    parser.add_argument("--normalize-unit-box", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    metadata_dir = out_dir / "metadata"
    assets_dir = out_dir / "assets"
    ply_dir = out_dir / "ply"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = metadata_dir / "annotated_800k.json"
    object_paths_path = metadata_dir / "object-paths.json.gz"
    lvis_annotations_path = metadata_dir / "lvis-annotations.json.gz"
    _download_if_missing(OBJAVERSE_PP_ANNOTATION_URL, annotation_path)
    _download_if_missing(OBJAVERSE_OBJECT_PATHS_URL, object_paths_path)

    annotations = _load_json(annotation_path)
    object_paths = _load_json_gz(object_paths_path)
    styles = _parse_set(args.styles)
    densities = _parse_set(args.densities)
    categories = _category_whitelist(args.category_preset, args.categories)
    uid_to_categories = None
    if categories:
        _download_if_missing(OBJAVERSE_LVIS_ANNOTATIONS_URL, lvis_annotations_path)
        uid_to_categories = _load_uid_to_categories(lvis_annotations_path, categories)

    candidates = _filter_candidates(
        annotations,
        object_paths,
        min_score=args.min_score,
        styles=styles,
        densities=densities,
        uid_to_categories=uid_to_categories,
    )
    selection = _select_candidates(
        candidates,
        start_index=args.start_index,
        count=args.count,
        per_category_count=args.per_category_count,
    )

    selection_manifest = out_dir / "selected_quality_manifest.json"
    _write_json(selection_manifest, selection)
    print(f"Selected {len(selection)} rows from {len(candidates)} strict candidates")
    print(f"Wrote selection manifest: {selection_manifest}")

    if args.manifest_only:
        return

    downloaded: list[dict[str, object]] = []
    for local_idx, row in enumerate(selection, start=args.start_index):
        uid = str(row["uid"])
        category = str(row.get("category", "object"))
        category_slug = _slugify(category)
        asset_relpath = str(row["asset_path"])
        suffix = Path(asset_relpath).suffix or ".glb"
        asset_path = assets_dir / f"{local_idx:06d}_{category_slug}_{uid}{suffix}"
        asset_url = f"{OBJAVERSE_ASSET_BASE_URL}/{asset_relpath}"

        try:
            if not asset_path.exists():
                print(f"Downloading {uid}: {asset_relpath}")
                _download_with_limit(asset_url, asset_path, args.max_asset_mb)

            mesh = _load_as_mesh(asset_path)
            xyz, rgb = _sample_mesh(mesh, args.points_per_object)
            color_std = float(np.asarray(rgb, dtype=np.float32).std(axis=0).mean())
            if args.require_color_variance and color_std < args.min_color_std:
                raise ValueError(f"sampled RGB variance is too low: mean channel std {color_std:.3f}")
            if args.normalize_unit_box:
                xyz = _normalize_unit_box(xyz)

            instance_id = np.full((xyz.shape[0],), len(downloaded) + 1, dtype=np.int32)
            semantic_id = np.full((xyz.shape[0],), int(args.semantic_id), dtype=np.int32)
            ply_path = ply_dir / f"{local_idx:06d}_{category_slug}_{uid}.ply"
            write_point_cloud_ply(ply_path, xyz, rgb, instance_id, semantic_id)

            record = dict(row)
            record.update(
                {
                    "local_index": local_idx,
                    "asset_url": asset_url,
                    "asset_local_path": str(asset_path),
                    "ply_path": str(ply_path),
                    "points": int(xyz.shape[0]),
                    "mean_rgb_channel_std": color_std,
                    "instance_id": int(len(downloaded) + 1),
                    "semantic_id": int(args.semantic_id),
                    "status": "ok",
                }
            )
            downloaded.append(record)
            print(f"Wrote {ply_path} ({xyz.shape[0]} points)")
            if args.target_success_count > 0:
                ok_count = sum(1 for item in downloaded if item.get("status") == "ok")
                if ok_count >= args.target_success_count:
                    print(f"Reached target success count: {args.target_success_count}")
                    break
        except Exception as exc:
            print(f"Skipping {uid}: {exc}")
            failed = dict(row)
            failed.update({"local_index": local_idx, "asset_url": asset_url, "status": "failed", "error": str(exc)})
            downloaded.append(failed)

    manifest_path = out_dir / "prepared_quality_manifest.json"
    _write_json(manifest_path, downloaded)
    ok_count = sum(1 for row in downloaded if row.get("status") == "ok")
    print(f"Wrote prepared manifest: {manifest_path}")
    print(f"Prepared {ok_count}/{len(downloaded)} assets")


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading metadata: {url}")
    urllib.request.urlretrieve(url, path)


def _download_with_limit(url: str, path: Path, max_mb: float) -> None:
    max_bytes = int(max_mb * 1024 * 1024)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise ValueError(f"asset is larger than limit: {int(length)} bytes > {max_bytes} bytes")
        total = 0
        with tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"asset exceeded limit while downloading: {total} bytes > {max_bytes} bytes")
                handle.write(chunk)
    tmp_path.replace(path)


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_json_gz(path: Path) -> object:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, rows: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _parse_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _category_whitelist(preset: str, categories: str) -> set[str] | None:
    selected: set[str] = set()
    preset_parts = _parse_set(preset)
    if "driving" in preset_parts:
        selected.update(DRIVING_LVIS_CATEGORIES)
    if "indoor" in preset_parts:
        selected.update(INDOOR_LVIS_CATEGORIES)
    selected.update(item.strip() for item in categories.split(",") if item.strip())
    return selected or None


def _load_uid_to_categories(path: Path, categories: set[str]) -> dict[str, list[str]]:
    lvis = _load_json_gz(path)
    if not isinstance(lvis, dict):
        raise TypeError("Objaverse LVIS annotations must be a dict")
    uid_to_categories: dict[str, list[str]] = {}
    for category in sorted(categories):
        for uid in lvis.get(category, []):
            uid_to_categories.setdefault(str(uid), []).append(category)
    return uid_to_categories


def _filter_candidates(
    annotations: object,
    object_paths: object,
    *,
    min_score: int,
    styles: set[str],
    densities: set[str],
    uid_to_categories: dict[str, list[str]] | None = None,
) -> list[dict[str, object]]:
    if not isinstance(annotations, list):
        raise TypeError("Objaverse++ annotations must be a list")
    if not isinstance(object_paths, dict):
        raise TypeError("Objaverse object paths must be a dict")

    candidates: list[dict[str, object]] = []
    for row in annotations:
        if not isinstance(row, dict):
            continue
        uid = str(row.get("UID") or row.get("uid") or "")
        asset_path = object_paths.get(uid)
        if not uid or not asset_path:
            continue
        categories = None
        if uid_to_categories is not None:
            categories = uid_to_categories.get(uid)
            if not categories:
                continue
        if int(row.get("score", -1)) < min_score:
            continue
        style = str(row.get("style", "")).lower()
        density = str(row.get("density", "")).lower()
        if styles and style not in styles:
            continue
        if densities and density not in densities:
            continue
        if any(_as_bool(row.get(key)) for key in ("is_multi_object", "is_scene", "is_transparent", "is_single_color")):
            continue
        asset_path = str(asset_path)
        if Path(asset_path).suffix.lower() not in {".glb", ".gltf"}:
            continue
        category = categories[0] if categories else "object"
        candidates.append(
            {
                "uid": uid,
                "category": category,
                "categories": categories or [],
                "score": int(row.get("score", -1)),
                "style": style,
                "density": density,
                "is_figure": str(row.get("is_figure", "false")).lower(),
                "asset_path": asset_path,
            }
        )

    density_rank = {"high": 0, "mid": 1, "low": 2}
    style_rank = {"scanned": 0, "realistic": 1, "cartoon": 2, "sci-fi": 3, "arcade": 4, "anime": 5, "other": 6}
    candidates.sort(
        key=lambda row: (
            str(row.get("category", "")),
            -int(row["score"]),
            density_rank.get(str(row["density"]), 99),
            style_rank.get(str(row["style"]), 99),
            str(row["uid"]),
        )
    )
    return candidates


def _select_candidates(
    candidates: list[dict[str, object]],
    *,
    start_index: int,
    count: int,
    per_category_count: int,
) -> list[dict[str, object]]:
    if per_category_count <= 0:
        return candidates[start_index : start_index + count]

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        grouped.setdefault(str(row.get("category", "object")), []).append(row)

    selection: list[dict[str, object]] = []
    for category in sorted(grouped):
        selection.extend(grouped[category][start_index : start_index + per_category_count])
    return selection[:count] if count > 0 else selection


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_")
    return slug[:80] or "object"


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


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

    offset = 12
    json_payload = None
    binary = None
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + chunk_len]
        if chunk_type == 0x4E4F534A:
            json_payload = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
        offset += 8 + chunk_len

    if json_payload is None or binary is None:
        return None
    if "KHR_draco_mesh_compression" not in json_payload.get("extensionsUsed", []):
        return None

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0
    for mesh_payload in json_payload.get("meshes", []):
        for primitive in mesh_payload.get("primitives", []):
            extension = primitive.get("extensions", {}).get("KHR_draco_mesh_compression")
            if extension is None:
                continue
            view = json_payload["bufferViews"][extension["bufferView"]]
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
