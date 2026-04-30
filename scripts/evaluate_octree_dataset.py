from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from threedvae.data.loaders.ply_loader import load_ply_frame
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.octree.tree import build_instance_octree
from threedvae.scene.builder import build_scene_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate octree statistics for a PLY object dataset.")
    parser.add_argument("--ply-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="dataset")
    parser.add_argument("--octree-preset", choices=("carla", "object"), default="carla")
    parser.add_argument("--token-leaf-only", action="store_true")
    parser.add_argument("--min-token-points", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_paths = sorted(Path(args.ply_dir).glob("*.ply"))
    if args.max_files > 0:
        ply_paths = ply_paths[: args.max_files]
    if not ply_paths:
        raise ValueError(f"No PLY files found in {args.ply_dir}.")

    config = build_octree_config(args.octree_preset)
    rows = []
    for path in ply_paths:
        rows.extend(
            evaluate_ply(
                path,
                config,
                token_leaf_only=args.token_leaf_only,
                min_token_points=args.min_token_points,
            )
        )

    summary = summarize(rows, label=args.label, ply_count=len(ply_paths))
    write_csv(out_dir / f"{args.label}_octree_stats.csv", rows)
    (out_dir / f"{args.label}_octree_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_ply(
    path: Path,
    config: OctreeBuildConfig,
    *,
    token_leaf_only: bool = False,
    min_token_points: int = 1,
) -> list[dict[str, object]]:
    frame = load_ply_frame(str(path))
    scene = build_scene_frame(frame)
    rows = []
    for instance in scene.instances:
        if instance.xyz_local is None or instance.xyz_local.shape[0] == 0:
            continue
        nodes = build_instance_octree(instance, config)
        token_nodes = [
            node
            for node in nodes
            if (not token_leaf_only or node.structure_state == "leaf")
            and len(node.point_indices) >= int(min_token_points)
        ]
        levels = [int(node.level) for node in nodes]
        leaf_nodes = [node for node in nodes if node.structure_state == "leaf"]
        split_flags = Counter(int(node.split_flag) for node in nodes)
        child_masks = Counter(int(node.child_mask) for node in nodes)
        node_point_counts = np.asarray([len(node.point_indices) for node in nodes], dtype=np.float32)
        leaf_point_counts = np.asarray([len(node.point_indices) for node in leaf_nodes], dtype=np.float32)
        size = instance.bbox_ego.size_xyz.astype(np.float32, copy=False)
        rows.append(
            {
                "ply": path.name,
                "instance_id": int(instance.instance_id),
                "semantic_id": int(instance.semantic_id),
                "category_name": str(instance.category_name),
                "points": int(instance.num_points),
                "tokens": int(len(nodes)),
                "filtered_tokens": int(len(token_nodes)),
                "nodes": int(len(nodes)),
                "leaf_nodes": int(len(leaf_nodes)),
                "internal_nodes": int(len(nodes) - len(leaf_nodes)),
                "max_depth": int(max(levels) if levels else 0),
                "mean_depth": float(np.mean(levels) if levels else 0.0),
                "leaf_ratio": float(len(leaf_nodes) / max(len(nodes), 1)),
                "mean_points_per_node": float(np.mean(node_point_counts) if node_point_counts.size else 0.0),
                "mean_points_per_leaf": float(np.mean(leaf_point_counts) if leaf_point_counts.size else 0.0),
                "min_points_per_leaf": int(np.min(leaf_point_counts) if leaf_point_counts.size else 0),
                "max_points_per_leaf": int(np.max(leaf_point_counts) if leaf_point_counts.size else 0),
                "bbox_x": float(size[0]),
                "bbox_y": float(size[1]),
                "bbox_z": float(size[2]),
                "bbox_volume": float(np.prod(np.maximum(size, 1e-6))),
                "split_flag_hist": dict(sorted(split_flags.items())),
                "child_mask_hist": dict(sorted(child_masks.items())),
            }
        )
    return rows


def summarize(rows: list[dict[str, object]], *, label: str, ply_count: int) -> dict[str, object]:
    tokens = np.asarray([row["tokens"] for row in rows], dtype=np.float32)
    points = np.asarray([row["points"] for row in rows], dtype=np.float32)
    max_depth = np.asarray([row["max_depth"] for row in rows], dtype=np.float32)
    leaf_ratio = np.asarray([row["leaf_ratio"] for row in rows], dtype=np.float32)
    semantic_hist = Counter(int(row["semantic_id"]) for row in rows)
    category_hist = Counter(str(row["category_name"]) for row in rows)
    return {
        "label": label,
        "ply_count": int(ply_count),
        "instance_count": int(len(rows)),
        "total_tokens": int(np.sum(tokens)) if tokens.size else 0,
        "total_filtered_tokens": int(np.sum(np.asarray([row["filtered_tokens"] for row in rows], dtype=np.float32))) if rows else 0,
        "tokens": describe(tokens),
        "points": describe(points),
        "max_depth": describe(max_depth),
        "leaf_ratio": describe(leaf_ratio),
        "semantic_hist": dict(sorted(semantic_hist.items())),
        "category_hist": dict(sorted(category_hist.items())),
    }


def describe(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "p25": 0.0, "mean": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "ply",
        "instance_id",
        "semantic_id",
        "category_name",
        "points",
        "tokens",
        "filtered_tokens",
        "nodes",
        "leaf_nodes",
        "internal_nodes",
        "max_depth",
        "mean_depth",
        "leaf_ratio",
        "mean_points_per_node",
        "mean_points_per_leaf",
        "min_points_per_leaf",
        "max_points_per_leaf",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "bbox_volume",
        "split_flag_hist",
        "child_mask_hist",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["split_flag_hist"] = json.dumps(serializable["split_flag_hist"], sort_keys=True)
            serializable["child_mask_hist"] = json.dumps(serializable["child_mask_hist"], sort_keys=True)
            writer.writerow(serializable)


def build_octree_config(preset: str) -> OctreeBuildConfig:
    if preset == "object":
        return OctreeBuildConfig.with_default_object_semantics()
    return OctreeBuildConfig.with_default_carla_semantics()


if __name__ == "__main__":
    main()
