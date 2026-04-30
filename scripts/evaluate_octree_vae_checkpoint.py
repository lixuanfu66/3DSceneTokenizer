from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from threedvae.data.dataset import build_node_dataset_from_ply_paths
from threedvae.models.octree_node_vae import OctreeNodeVAE
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.utils.torch_compat import DataLoader, require_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an octree node VAE checkpoint on PLY data.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ply-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--octree-preset", choices=("carla", "object"), default="carla")
    parser.add_argument("--points-per-node", type=int, default=128)
    parser.add_argument("--queries-per-node", type=int, default=128)
    parser.add_argument("--query-strategy", choices=("uniform", "layered", "calibrated_surface"), default="uniform")
    parser.add_argument("--include-leaf-only", action="store_true")
    parser.add_argument("--min-node-points", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    torch, _, _, _, _ = require_torch()
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_paths = sorted(str(path) for path in Path(args.ply_dir).glob("*.ply"))
    dataset = build_node_dataset_from_ply_paths(
        ply_paths,
        points_per_node=args.points_per_node,
        queries_per_node=args.queries_per_node,
        seed=args.seed,
        octree_config=build_octree_config(args.octree_preset),
        include_leaf_only=args.include_leaf_only,
        min_node_points=args.min_node_points,
        query_strategy=args.query_strategy,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.torch_collate)

    payload = torch.load(args.checkpoint, map_location=args.device)
    model_config = payload.get("model_config")
    if not model_config:
        raise ValueError("Checkpoint is missing model_config.")
    model = OctreeNodeVAE(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(args.device)
    model.eval()

    totals = {
        "query_count": 0,
        "udf_abs_sum": 0.0,
        "udf_sq_sum": 0.0,
        "udf_max_abs": 0.0,
        "occ_correct": 0,
        "occ_count": 0,
        "rgb_abs_sum": 0.0,
        "rgb_sq_sum": 0.0,
        "rgb_count": 0,
    }
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in batch.items()}
            outputs = model(
                xyz=moved["xyz"],
                rgb=moved["rgb"],
                query_xyz=moved["query_xyz"],
                rgb_query_xyz=moved.get("query_rgb_xyz") if model.config.predict_rgb else None,
                node_center=moved["node_center_local"],
                node_size=moved["node_size_local"],
                level=moved["level"],
                split_flag=moved["split_flag"],
                child_index=moved["child_index"],
                semantic_id=moved["semantic_id"],
            )
            udf_error = (outputs.udf.detach() - moved["query_udf"]).abs()
            totals["query_count"] += int(udf_error.numel())
            totals["udf_abs_sum"] += float(udf_error.sum().detach().cpu())
            totals["udf_sq_sum"] += float((udf_error * udf_error).sum().detach().cpu())
            totals["udf_max_abs"] = max(totals["udf_max_abs"], float(udf_error.max().detach().cpu()))

            pred_occ = torch.sigmoid(outputs.occ_logits.detach()) >= 0.5
            target_occ = moved["query_occ"] >= 0.5
            totals["occ_correct"] += int((pred_occ == target_occ).sum().detach().cpu())
            totals["occ_count"] += int(pred_occ.numel())

            if outputs.rgb is not None and "query_rgb" in moved:
                rgb_error = (outputs.rgb.detach() - moved["query_rgb"]).abs()
                totals["rgb_abs_sum"] += float(rgb_error.sum().detach().cpu())
                totals["rgb_sq_sum"] += float((rgb_error * rgb_error).sum().detach().cpu())
                totals["rgb_count"] += int(rgb_error.numel())

    query_count = max(int(totals["query_count"]), 1)
    rgb_count = max(int(totals["rgb_count"]), 1)
    rgb_mse = totals["rgb_sq_sum"] / rgb_count if totals["rgb_count"] else None
    metrics = {
        "checkpoint": args.checkpoint,
        "ply_count": len(ply_paths),
        "include_leaf_only": bool(args.include_leaf_only),
        "min_node_points": int(args.min_node_points),
        "sample_count": len(dataset),
        "query_count": int(totals["query_count"]),
        "udf_mae": totals["udf_abs_sum"] / query_count,
        "udf_rmse": math.sqrt(totals["udf_sq_sum"] / query_count),
        "udf_max_abs": totals["udf_max_abs"],
        "occ_accuracy": totals["occ_correct"] / max(int(totals["occ_count"]), 1),
        "rgb_mae": (totals["rgb_abs_sum"] / rgb_count) if totals["rgb_count"] else None,
        "rgb_mse": rgb_mse,
        "rgb_psnr": (-10.0 * math.log10(max(rgb_mse, 1e-12))) if rgb_mse is not None else None,
    }
    (out_dir / "checkpoint_eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_octree_config(preset: str) -> OctreeBuildConfig:
    if preset == "object":
        return OctreeBuildConfig.with_default_object_semantics()
    return OctreeBuildConfig.with_default_carla_semantics()


if __name__ == "__main__":
    main()
