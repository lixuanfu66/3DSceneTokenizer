from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from threedvae.data.bench2drive_rgbd import (
    camera_calibration_from_annotation,
    filter_point_cloud_bounds,
    infer_bench2drive_paths,
    load_annotation,
    rgbd_to_point_cloud,
    rgbd_to_ply,
    write_point_cloud_ply,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project one Bench2Drive RGB-D camera frame to a colored PLY point cloud with instance and semantic ids."
    )
    parser.add_argument("--route-dir", help="Bench2Drive route directory, e.g. .../AccidentTwoWays/Town12_Route1444_Weather0.")
    parser.add_argument("--camera", default="front", help="Camera view: front, front_left, front_right, back, back_left, back_right, or CAM_*.")
    parser.add_argument("--frame", help="Frame id such as 00000. Used with --route-dir to infer input paths.")
    parser.add_argument("--rgb", help="Explicit RGB image path.")
    parser.add_argument("--depth", help="Explicit depth image path.")
    parser.add_argument("--semantic", help="Explicit semantic segmentation image path.")
    parser.add_argument("--instance", help="Explicit instance segmentation image path.")
    parser.add_argument("--annotation", help="Annotation JSON or JSON.GZ path containing sensors/CAM_*/intrinsic/cam2ego/world2cam.")
    parser.add_argument("--intrinsic-json", help="JSON file containing a 3x3 intrinsic matrix. Overrides --annotation intrinsic.")
    parser.add_argument("--cam2ego-json", help="JSON file containing a 4x4 cam2ego matrix. Overrides --annotation cam2ego.")
    parser.add_argument("--world2cam-json", help="JSON file containing a 4x4 world2cam matrix. Overrides --annotation world2cam.")
    parser.add_argument("--out", required=True, help="Output PLY path.")
    parser.add_argument("--target-frame", choices=["camera", "ego", "world"], default="ego")
    parser.add_argument("--depth-encoding", choices=["carla-rgb", "meters", "uint16-mm", "uint16-cm", "uint16-dm"], default="carla-rgb")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Multiply decoded depth by this value before projection.")
    parser.add_argument("--depth-offset", type=float, default=0.0, help="Add this value in meters after depth scaling.")
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
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth pixel to reduce output size.")
    parser.add_argument("--max-depth-m", type=float, default=None, help="Optional depth clipping in meters.")
    parser.add_argument("--min-x", type=float, default=None)
    parser.add_argument("--max-x", type=float, default=None)
    parser.add_argument("--min-y", type=float, default=None)
    parser.add_argument("--max-y", type=float, default=None)
    parser.add_argument("--min-z", type=float, default=None)
    parser.add_argument("--max-z", type=float, default=None)
    args = parser.parse_args()

    paths = _resolve_paths(args)
    intrinsic, cam2ego, world2cam = _resolve_calibration(args, paths.get("annotation"))
    include_semantics = _parse_int_set(args.depth_dequantization_include_semantics)

    if any(value is not None for value in (args.min_x, args.max_x, args.min_y, args.max_y, args.min_z, args.max_z)):
        xyz, rgb, instance, semantic = rgbd_to_point_cloud(
            rgb_path=paths["rgb"],
            depth_path=paths["depth"],
            semantic_path=paths["semantic"],
            instance_path=paths["instance"],
            intrinsic=intrinsic,
            cam2ego=cam2ego,
            world2cam=world2cam,
            target_frame=args.target_frame,
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
        count = int(xyz.shape[0])
    else:
        count = rgbd_to_ply(
            rgb_path=paths["rgb"],
            depth_path=paths["depth"],
            semantic_path=paths["semantic"],
            instance_path=paths["instance"],
            out_path=args.out,
            intrinsic=intrinsic,
            cam2ego=cam2ego,
            world2cam=world2cam,
            target_frame=args.target_frame,
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
    print(f"Wrote {count} points to {args.out}")


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    inferred: dict[str, Path] = {}
    if args.route_dir:
        if not args.frame:
            raise ValueError("--frame is required when --route-dir is used.")
        inferred = infer_bench2drive_paths(args.route_dir, args.camera, args.frame)

    paths = {
        "rgb": Path(args.rgb) if args.rgb else inferred.get("rgb"),
        "depth": Path(args.depth) if args.depth else inferred.get("depth"),
        "semantic": Path(args.semantic) if args.semantic else inferred.get("semantic"),
        "instance": Path(args.instance) if args.instance else inferred.get("instance"),
        "annotation": Path(args.annotation) if args.annotation else inferred.get("annotation"),
    }
    missing = [key for key in ("rgb", "depth", "semantic", "instance") if paths[key] is None]
    if missing:
        raise ValueError(f"Missing required input paths: {missing}. Provide --route-dir/--frame or explicit image paths.")
    for key, path in paths.items():
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{key} path does not exist: {path}")
    return paths  # type: ignore[return-value]


def _resolve_calibration(args: argparse.Namespace, annotation_path: Path | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    intrinsic = _load_json_matrix(args.intrinsic_json) if args.intrinsic_json else None
    cam2ego = _load_json_matrix(args.cam2ego_json) if args.cam2ego_json else None
    world2cam = _load_json_matrix(args.world2cam_json) if args.world2cam_json else None

    needs_cam2ego = args.target_frame == "ego" and cam2ego is None
    needs_world2cam = args.target_frame == "world" and world2cam is None
    if intrinsic is None or needs_cam2ego or needs_world2cam:
        if annotation_path is None:
            raise ValueError("Calibration requires --annotation, or explicit --intrinsic-json plus the transform needed by --target-frame.")
        anno_intrinsic, anno_cam2ego, anno_world2cam = camera_calibration_from_annotation(load_annotation(annotation_path), args.camera)
        intrinsic = anno_intrinsic if intrinsic is None else intrinsic
        cam2ego = anno_cam2ego if needs_cam2ego else cam2ego
        world2cam = anno_world2cam if needs_world2cam else world2cam

    if intrinsic.shape != (3, 3):
        raise ValueError(f"Intrinsic must be 3x3, got {intrinsic.shape}.")
    return intrinsic, cam2ego, world2cam


def _load_json_matrix(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return np.asarray(json.load(handle), dtype=np.float64)


def _parse_int_set(raw: str) -> set[int] | None:
    if not raw.strip():
        return None
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


if __name__ == "__main__":
    main()
