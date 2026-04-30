from __future__ import annotations

import argparse
import json
from pathlib import Path

from threedvae.data.loaders.ply_loader import load_ply_frame
from threedvae.debug.exporters import export_scene_octree_node_bboxes_ply
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.octree.tree import build_octree_debug_records
from threedvae.scene.builder import build_scene_frame
from threedvae.tokenizer.instance_encoder import encode_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export octree node bbox PLY files for a PLY directory.")
    parser.add_argument("--ply-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--octree-preset", choices=("carla", "object"), default="carla")
    parser.add_argument("--leaf-only", action="store_true")
    parser.add_argument("--occupied-only", action="store_true")
    parser.add_argument("--min-node-points", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = build_octree_config(args.octree_preset)
    ply_paths = sorted(Path(args.ply_dir).glob("*.ply"))
    if args.max_files > 0:
        ply_paths = ply_paths[: args.max_files]

    summary = {
        "ply_dir": args.ply_dir,
        "octree_preset": args.octree_preset,
        "leaf_only": bool(args.leaf_only),
        "occupied_only": bool(args.occupied_only),
        "min_node_points": int(args.min_node_points),
        "frames": {},
    }
    for ply_path in ply_paths:
        frame = load_ply_frame(str(ply_path))
        scene = build_scene_frame(frame)
        scene_node_debug_records = []
        total_nodes = 0
        for instance in scene.instances:
            encoded = encode_instance(instance, octree_config=config)
            debug_records = build_octree_debug_records(instance, encoded.octree_nodes)
            debug_records = filter_records(
                debug_records,
                leaf_only=args.leaf_only,
                occupied_only=args.occupied_only,
                min_node_points=args.min_node_points,
            )
            scene_node_debug_records.append((instance, debug_records))
            total_nodes += len(debug_records)
        out_name = f"{safe_name(scene.frame_id)}_octree_node_bboxes.ply"
        export_scene_octree_node_bboxes_ply(scene_node_debug_records, str(out_dir / out_name))
        summary["frames"][scene.frame_id] = {
            "instances": len(scene.instances),
            "total_nodes": total_nodes,
            "ply": out_name,
        }

    summary["ply_count"] = len(ply_paths)
    summary["total_nodes"] = sum(item["total_nodes"] for item in summary["frames"].values())
    (out_dir / "octree_node_bbox_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_octree_config(preset: str) -> OctreeBuildConfig:
    if preset == "object":
        return OctreeBuildConfig.with_default_object_semantics()
    return OctreeBuildConfig.with_default_carla_semantics()


def filter_records(records, *, leaf_only: bool, occupied_only: bool, min_node_points: int):
    filtered = []
    for record in records:
        if leaf_only and record.structure_state != "leaf":
            continue
        if occupied_only and int(record.num_points) <= 0:
            continue
        if int(record.num_points) < int(min_node_points):
            continue
        filtered.append(record)
    return filtered


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


if __name__ == "__main__":
    main()
