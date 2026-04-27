from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from threedvae.data.bench2drive_rgbd import (
    camera_calibration_from_annotation,
    filter_point_cloud_bounds,
    infer_bench2drive_paths,
    load_annotation,
    rgbd_to_point_cloud,
    write_point_cloud_ply,
)


DEFAULT_CAMERAS = ("front", "front_left", "front_right", "back", "back_left", "back_right")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project multiple Bench2Drive RGB-D camera views into one ego-frame colored PLY."
    )
    parser.add_argument("--route-dir", required=True, help="Bench2Drive route directory.")
    parser.add_argument("--frame", required=True, help="Frame id such as 00000.")
    parser.add_argument("--out", required=True, help="Output fused PLY path.")
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated camera views. Missing modalities are skipped when --skip-missing is enabled.",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip cameras without RGB/depth/semantic/instance/annotation files.")
    parser.add_argument("--depth-encoding", choices=["carla-rgb", "meters", "uint16-mm", "uint16-cm", "uint16-dm"], default="meters")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--depth-offset", type=float, default=0.0)
    parser.add_argument("--depth-dequantization", choices=["none", "column-ramp", "column-ramp-reverse", "column-ramp-auto"], default="none")
    parser.add_argument("--depth-dequantization-min-run", type=int, default=2)
    parser.add_argument("--depth-dequantization-semantic-aware", action="store_true")
    parser.add_argument(
        "--depth-dequantization-include-semantics",
        default="",
        help="Comma-separated semantic ids to dequantize. Empty means all semantics.",
    )
    parser.add_argument("--semantic-encoding", choices=["auto", "red", "green", "blue", "gray", "rgb24"], default="auto")
    parser.add_argument("--instance-encoding", choices=["auto", "red", "green", "blue", "gray", "rgb24"], default="rgb24")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-depth-m", type=float, default=120.0)
    parser.add_argument("--min-x", type=float, default=None)
    parser.add_argument("--max-x", type=float, default=None)
    parser.add_argument("--min-y", type=float, default=None)
    parser.add_argument("--max-y", type=float, default=None)
    parser.add_argument("--min-z", type=float, default=None)
    parser.add_argument("--max-z", type=float, default=None)
    args = parser.parse_args()

    route_dir = Path(args.route_dir)
    cameras = _parse_cameras(args.cameras)
    annotation_path = route_dir / "anno" / f"{Path(args.frame).stem}.json.gz"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
    annotation = load_annotation(annotation_path)
    include_semantics = _parse_int_set(args.depth_dequantization_include_semantics)

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    instance_parts: list[np.ndarray] = []
    semantic_parts: list[np.ndarray] = []
    used_cameras: list[str] = []

    for camera in cameras:
        paths = infer_bench2drive_paths(route_dir, camera, args.frame)
        missing = [key for key in ("rgb", "depth", "semantic", "instance") if not paths[key].exists()]
        if missing:
            message = f"Skipping {camera}: missing {', '.join(missing)}"
            if not args.skip_missing:
                raise FileNotFoundError(message)
            print(message)
            continue

        intrinsic, cam2ego, _world2cam = camera_calibration_from_annotation(annotation, camera)
        points, rgb, instance, semantic = rgbd_to_point_cloud(
            rgb_path=paths["rgb"],
            depth_path=paths["depth"],
            semantic_path=paths["semantic"],
            instance_path=paths["instance"],
            intrinsic=intrinsic,
            cam2ego=cam2ego,
            target_frame="ego",
            depth_encoding=args.depth_encoding,
            semantic_encoding=args.semantic_encoding,
            instance_encoding=args.instance_encoding,
            stride=args.stride,
            max_depth_m=args.max_depth_m,
            depth_scale=args.depth_scale,
            depth_offset=args.depth_offset,
            depth_dequantization=args.depth_dequantization,
            depth_dequantization_min_run=args.depth_dequantization_min_run,
            depth_dequantization_semantic_aware=args.depth_dequantization_semantic_aware,
            depth_dequantization_include_semantics=include_semantics,
        )
        xyz_parts.append(points)
        rgb_parts.append(rgb)
        instance_parts.append(instance)
        semantic_parts.append(semantic)
        used_cameras.append(camera)
        print(f"{camera}: {points.shape[0]} points")

    if not xyz_parts:
        raise ValueError("No camera produced any points.")

    xyz = np.concatenate(xyz_parts, axis=0)
    rgb = np.concatenate(rgb_parts, axis=0)
    instance = np.concatenate(instance_parts, axis=0)
    semantic = np.concatenate(semantic_parts, axis=0)
    xyz, rgb, instance, semantic = filter_point_cloud_bounds(
        xyz,
        rgb,
        instance,
        semantic,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
        min_z=args.min_z,
        max_z=args.max_z,
    )
    write_point_cloud_ply(args.out, xyz, rgb, instance, semantic)
    print(f"Wrote {xyz.shape[0]} fused points from {used_cameras} to {args.out}")


def _parse_cameras(raw: str) -> list[str]:
    cameras = [item.strip() for item in raw.split(",") if item.strip()]
    if not cameras:
        raise ValueError("--cameras must contain at least one camera.")
    return cameras


def _parse_int_set(raw: str) -> set[int] | None:
    if not raw.strip():
        return None
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


if __name__ == "__main__":
    main()
