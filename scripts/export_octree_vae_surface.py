from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from threedvae.data.dataset import collect_node_samples_from_ply_paths, sample_point_cloud
from threedvae.data.loaders.ply_loader import load_ply_frame
from threedvae.models.octree_node_vae import OctreeNodeVAE
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.scene.builder import build_scene_frame
from threedvae.utils.torch_compat import require_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export predicted surface points from an octree node VAE.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ply-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--points-per-node", type=int, default=32)
    parser.add_argument("--candidates-per-node", type=int, default=512)
    parser.add_argument("--candidate-strategy", choices=("uniform", "hybrid"), default="hybrid")
    parser.add_argument("--surface-threshold", type=float, default=0.03)
    parser.add_argument("--topk-per-node", type=int, default=4)
    parser.add_argument("--topk-max-udf", type=float, default=0.08)
    parser.add_argument("--batch-nodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-points-per-frame", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    torch, _, _, _, _ = require_torch()
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_paths = sorted(str(path) for path in Path(args.ply_dir).glob("*.ply"))
    octree_config = OctreeBuildConfig.with_default_carla_semantics()
    samples = collect_node_samples_from_ply_paths(ply_paths, octree_config=octree_config)
    pose_lookup = build_pose_lookup(ply_paths)

    payload = torch.load(args.checkpoint, map_location=args.device)
    model = OctreeNodeVAE(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(args.device)
    model.eval()

    frame_rows: dict[str, list[tuple[float, float, float, int, int, int, int, float, int]]] = defaultdict(list)
    frame_stats: dict[str, dict[str, float | int]] = defaultdict(empty_stats)

    for start in range(0, len(samples), args.batch_nodes):
        batch_samples = samples[start : start + args.batch_nodes]
        batch = build_reconstruction_batch(batch_samples, args)
        with torch.no_grad():
            outputs = model(
                xyz=torch.as_tensor(batch["xyz"], dtype=torch.float32, device=args.device),
                rgb=torch.as_tensor(batch["rgb"], dtype=torch.float32, device=args.device),
                query_xyz=torch.as_tensor(batch["query_xyz"], dtype=torch.float32, device=args.device),
                node_center=torch.as_tensor(batch["node_center_local"], dtype=torch.float32, device=args.device),
                node_size=torch.as_tensor(batch["node_size_local"], dtype=torch.float32, device=args.device),
                level=torch.as_tensor(batch["level"], dtype=torch.int64, device=args.device),
                split_flag=torch.as_tensor(batch["split_flag"], dtype=torch.int64, device=args.device),
                child_index=torch.as_tensor(batch["child_index"], dtype=torch.int64, device=args.device),
                semantic_id=torch.as_tensor(batch["semantic_id"], dtype=torch.int64, device=args.device),
            )
        pred_udf = outputs.udf.detach().cpu().numpy()[..., 0]
        for item_index, sample in enumerate(batch_samples):
            frame_id = sample.frame_id
            stats = frame_stats[frame_id]
            stats["total_nodes"] = int(stats["total_nodes"]) + 1
            stats["total_candidates"] = int(stats["total_candidates"]) + int(args.candidates_per_node)
            query_local = batch["query_xyz"][item_index]
            keep_mask, keep_kind = select_surface_candidates(
                pred_udf[item_index],
                threshold=args.surface_threshold,
                topk_per_node=args.topk_per_node,
                topk_max_udf=args.topk_max_udf,
            )
            stats["threshold_points"] = int(stats["threshold_points"]) + int(np.sum(pred_udf[item_index] <= args.surface_threshold))
            stats["kept_points"] = int(stats["kept_points"]) + int(np.sum(keep_mask))
            if not np.any(keep_mask):
                continue
            pose = pose_lookup[(frame_id, int(sample.instance_id))]
            query_ego = local_to_ego(query_local[keep_mask], pose)
            pred_kept = pred_udf[item_index][keep_mask]
            kind_kept = keep_kind[keep_mask]
            for point, pred_value, keep_value in zip(query_ego, pred_kept, kind_kept):
                color = color_from_udf(float(pred_value), args.surface_threshold, args.topk_max_udf)
                frame_rows[frame_id].append(
                    (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        color[0],
                        color[1],
                        color[2],
                        int(sample.instance_id),
                        float(pred_value),
                        int(sample.node_id),
                        int(sample.semantic_id),
                        int(keep_value),
                    )
                )

    summary = {
        "checkpoint": args.checkpoint,
        "ply_count": len(ply_paths),
        "candidate_strategy": args.candidate_strategy,
        "points_per_node": args.points_per_node,
        "candidates_per_node": args.candidates_per_node,
        "surface_threshold": args.surface_threshold,
        "topk_per_node": args.topk_per_node,
        "topk_max_udf": args.topk_max_udf,
        "frames": {},
    }
    rng = np.random.default_rng(args.seed + 99_999)
    for frame_id in sorted(frame_rows):
        rows = frame_rows[frame_id]
        if args.max_points_per_frame > 0 and len(rows) > args.max_points_per_frame:
            choice = rng.choice(len(rows), size=args.max_points_per_frame, replace=False)
            rows = [rows[int(index)] for index in choice]
        out_name = f"{safe_name(frame_id)}_pred_surface_{args.candidate_strategy}_udf{threshold_tag(args.surface_threshold)}.ply"
        write_surface_ply(out_dir / out_name, rows)
        stats = frame_stats[frame_id]
        total_candidates = max(int(stats["total_candidates"]), 1)
        summary["frames"][frame_id] = {
            "total_tokens": int(stats["total_nodes"]),
            "total_nodes": int(stats["total_nodes"]),
            "total_candidates": int(stats["total_candidates"]),
            "threshold_points": int(stats["threshold_points"]),
            "kept_points": int(stats["kept_points"]),
            "retention_ratio": float(stats["kept_points"]) / total_candidates,
            "ply": out_name,
        }
    summary["total_tokens"] = sum(item["total_tokens"] for item in summary["frames"].values())
    summary["total_candidates"] = sum(item["total_candidates"] for item in summary["frames"].values())
    summary["total_kept_points"] = sum(item["kept_points"] for item in summary["frames"].values())
    (out_dir / "surface_export_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_reconstruction_batch(samples, args: argparse.Namespace) -> dict[str, np.ndarray]:
    xyz_batch = []
    rgb_batch = []
    query_batch = []
    centers = []
    sizes = []
    levels = []
    split_flags = []
    child_indices = []
    semantic_ids = []
    for offset, sample in enumerate(samples):
        xyz, rgb = sample_point_cloud(
            sample.xyz_local,
            sample.rgb,
            num_points=args.points_per_node,
            seed=args.seed + offset + int(sample.node_id),
        )
        query_xyz = build_candidate_queries(
            sample.xyz_local,
            node_center=sample.center_local,
            node_size=sample.size_local,
            split_flag=sample.split_flag,
            count=args.candidates_per_node,
            strategy=args.candidate_strategy,
            seed=args.seed + 1_000_000 + offset + int(sample.node_id),
        )
        xyz_batch.append(xyz.astype(np.float32, copy=False))
        rgb_batch.append(rgb.astype(np.float32) / 255.0)
        query_batch.append(query_xyz.astype(np.float32, copy=False))
        centers.append(sample.center_local.astype(np.float32, copy=False))
        sizes.append(sample.size_local.astype(np.float32, copy=False))
        levels.append(int(sample.level))
        split_flags.append(int(sample.split_flag))
        child_indices.append(-1 if sample.child_index is None else int(sample.child_index))
        semantic_ids.append(int(sample.semantic_id))
    return {
        "xyz": np.stack(xyz_batch, axis=0),
        "rgb": np.stack(rgb_batch, axis=0),
        "query_xyz": np.stack(query_batch, axis=0),
        "node_center_local": np.stack(centers, axis=0),
        "node_size_local": np.stack(sizes, axis=0),
        "level": np.asarray(levels, dtype=np.int64),
        "split_flag": np.asarray(split_flags, dtype=np.int64),
        "child_index": np.asarray(child_indices, dtype=np.int64),
        "semantic_id": np.asarray(semantic_ids, dtype=np.int64),
    }


def build_candidate_queries(
    xyz: np.ndarray,
    *,
    node_center: np.ndarray,
    node_size: np.ndarray,
    split_flag: int,
    count: int,
    strategy: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    node_size = np.maximum(node_size.astype(np.float32, copy=False), 1e-2)
    node_center = node_center.astype(np.float32, copy=False)
    bbox_min = node_center - 0.5 * node_size
    bbox_max = node_center + 0.5 * node_size
    if strategy == "uniform":
        return rng.uniform(bbox_min, bbox_max, size=(count, 3)).astype(np.float32)

    uniform_count = int(round(count * 0.45))
    observed_count = int(round(count * 0.4))
    split_count = max(0, count - uniform_count - observed_count)
    uniform = rng.uniform(bbox_min, bbox_max, size=(uniform_count, 3)).astype(np.float32)
    choice = rng.choice(xyz.shape[0], size=observed_count, replace=xyz.shape[0] < observed_count)
    sigma = np.maximum(node_size * 0.015, 0.005).astype(np.float32)
    observed = xyz[choice].astype(np.float32, copy=False) + rng.normal(
        loc=0.0,
        scale=sigma[None, :],
        size=(observed_count, 3),
    ).astype(np.float32)
    split = rng.uniform(bbox_min, bbox_max, size=(split_count, 3)).astype(np.float32)
    active_axes = [axis for axis in range(3) if int(split_flag) & (1 << axis)] or [0, 1, 2]
    plane_sigma = np.maximum(node_size * 0.01, 1e-3)
    for row in split:
        axis = int(rng.choice(active_axes))
        row[axis] = node_center[axis] + float(rng.normal(0.0, plane_sigma[axis]))
    queries = np.concatenate([uniform, observed, split], axis=0)
    if queries.shape[0] < count:
        extra = rng.uniform(bbox_min, bbox_max, size=(count - queries.shape[0], 3)).astype(np.float32)
        queries = np.concatenate([queries, extra], axis=0)
    return queries[:count].astype(np.float32, copy=False)


def select_surface_candidates(
    pred_udf: np.ndarray,
    *,
    threshold: float,
    topk_per_node: int,
    topk_max_udf: float,
) -> tuple[np.ndarray, np.ndarray]:
    keep = pred_udf <= float(threshold)
    keep_kind = np.where(keep, 1, 0).astype(np.int32)
    if topk_per_node > 0:
        topk = min(int(topk_per_node), pred_udf.shape[0])
        indices = np.argpartition(pred_udf, kth=topk - 1)[:topk]
        topk_mask = np.zeros_like(keep)
        topk_mask[indices] = pred_udf[indices] <= float(topk_max_udf)
        keep_kind[topk_mask & ~keep] = 2
        keep = keep | topk_mask
    return keep, keep_kind


def build_pose_lookup(ply_paths: list[str]) -> dict[tuple[str, int], tuple[np.ndarray, float]]:
    lookup = {}
    for path in ply_paths:
        frame = load_ply_frame(path)
        scene = build_scene_frame(frame)
        for instance in scene.instances:
            lookup[(scene.frame_id, int(instance.instance_id))] = (
                instance.pose_ego.center_xyz.astype(np.float32, copy=False),
                float(instance.pose_ego.yaw),
            )
    return lookup


def local_to_ego(xyz_local: np.ndarray, pose: tuple[np.ndarray, float]) -> np.ndarray:
    center, yaw = pose
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (xyz_local.astype(np.float32, copy=False) @ rotation.T) + center[None, :]


def color_from_udf(pred_udf: float, threshold: float, topk_max_udf: float) -> tuple[int, int, int]:
    if pred_udf <= threshold:
        return 30, 245, 80
    scale = min(1.0, max(0.0, (pred_udf - threshold) / max(topk_max_udf - threshold, 1e-6)))
    return int(round(255 * scale)), int(round(220 * (1.0 - scale))), 60


def empty_stats() -> dict[str, float | int]:
    return {
        "total_nodes": 0,
        "total_candidates": 0,
        "threshold_points": 0,
        "kept_points": 0,
    }


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def threshold_tag(value: float) -> str:
    return str(value).replace(".", "p")


def write_surface_ply(
    path: Path,
    rows: list[tuple[float, float, float, int, int, int, int, float, int, int, int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(rows)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property int instance_id\n")
        handle.write("property float pred_udf\n")
        handle.write("property int node_id\n")
        handle.write("property int semantic_id\n")
        handle.write("property int keep_kind\n")
        handle.write("comment keep_kind 1=threshold 2=topk_fallback\n")
        handle.write("end_header\n")
        for x, y, z, red, green, blue, instance_id, pred_udf, node_id, semantic_id, keep_kind in rows:
            handle.write(
                f"{x:.6f} {y:.6f} {z:.6f} {red} {green} {blue} "
                f"{instance_id} {pred_udf:.6f} {node_id} {semantic_id} {keep_kind}\n"
            )


if __name__ == "__main__":
    main()
