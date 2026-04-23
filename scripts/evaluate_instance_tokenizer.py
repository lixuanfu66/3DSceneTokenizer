from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from threedvae.data.dataset import (
    build_instance_dataset_from_ply_paths,
    build_node_dataset_from_ply_paths,
)
from threedvae.eval.metrics import (
    batch_reconstruction_metrics,
    summarize_codebook_usage,
    summarize_compression,
)
from threedvae.eval.reporting import EvaluationReport, write_evaluation_bundle
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.models.instance_tokenizer import PointNetVQTokenizer
from threedvae.utils.torch_compat import DataLoader, require_torch, torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained instance/node tokenizer checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path from train_instance_tokenizer.py.")
    parser.add_argument("--ply-dir", default=None, help="Directory containing evaluation PLY files.")
    parser.add_argument("--out", required=True, help="Directory to write evaluation outputs.")
    parser.add_argument("--sample-unit", choices=("instance", "node"), default="node", help="Evaluation sample unit.")
    parser.add_argument("--points-per-instance", type=int, default=256, help="Sampled point count in instance mode.")
    parser.add_argument("--points-per-node", type=int, default=128, help="Sampled point count in node mode.")
    parser.add_argument("--queries-per-node", type=int, default=64, help="UDF query count in node mode.")
    parser.add_argument("--udf-truncation", type=float, default=0.25, help="UDF truncation distance in node mode.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Evaluation device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--include-leaf-only", action="store_true", help="Only evaluate leaf nodes in node mode.")
    parser.add_argument("--high-priority-semantics", default="", help="Comma-separated semantic ids forced to XYZ split.")
    parser.add_argument("--xy-only-semantics", default="", help="Comma-separated semantic ids forced to XY split.")
    parser.add_argument("--xz-only-semantics", default="", help="Comma-separated semantic ids forced to XZ split.")
    parser.add_argument("--yz-only-semantics", default="", help="Comma-separated semantic ids forced to YZ split.")
    return parser.parse_args()


def main() -> None:
    require_torch()
    args = parse_args()
    if not args.ply_dir:
        raise ValueError("You must provide --ply-dir for evaluation.")

    eval_paths = sorted(str(path) for path in Path(args.ply_dir).glob("*.ply"))
    if not eval_paths:
        raise ValueError("No PLY files found in --ply-dir.")

    checkpoint_payload = torch.load(args.checkpoint, map_location=args.device)
    model_config = checkpoint_payload.get("model_config")
    if not model_config:
        raise ValueError("Checkpoint is missing model_config; retrain with the current trainer.")

    model = PointNetVQTokenizer(**model_config)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    octree_config = OctreeBuildConfig(
        high_priority_semantics=_parse_semantic_ids(args.high_priority_semantics),
        xy_only_semantics=_parse_semantic_ids(args.xy_only_semantics),
        xz_only_semantics=_parse_semantic_ids(args.xz_only_semantics),
        yz_only_semantics=_parse_semantic_ids(args.yz_only_semantics),
    )
    if args.sample_unit == "node":
        dataset = build_node_dataset_from_ply_paths(
            eval_paths,
            points_per_node=args.points_per_node,
            queries_per_node=args.queries_per_node,
            udf_truncation=args.udf_truncation,
            seed=args.seed,
            octree_config=octree_config,
            include_leaf_only=args.include_leaf_only,
        )
    else:
        dataset = build_instance_dataset_from_ply_paths(
            eval_paths,
            points_per_instance=args.points_per_instance,
            seed=args.seed,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.torch_collate,
    )

    accumulated = {
        "xyz_mse": 0.0,
        "rgb_mse": 0.0,
        "chamfer_l2": 0.0,
        "udf_smooth_l1": 0.0,
        "metric_steps": 0,
    }
    code_indices: list[int] = []
    sample_lengths = [int(sample.xyz_local.shape[0]) for sample in dataset.samples]
    udf_metric_steps = 0

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(args.device)
            target_xyz = batch["xyz"].to(args.device)
            target_rgb = batch["rgb"].to(args.device)
            query_xyz = batch["query_xyz"].to(args.device) if "query_xyz" in batch else None
            target_udf = batch["query_udf"].to(args.device) if "query_udf" in batch else None
            outputs = model(points, query_xyz=query_xyz)
            metrics = batch_reconstruction_metrics(
                outputs.recon_points,
                target_xyz,
                target_rgb,
                udf_logits=outputs.udf_logits,
                target_udf=target_udf,
            )
            accumulated["xyz_mse"] += metrics.xyz_mse
            accumulated["rgb_mse"] += metrics.rgb_mse
            accumulated["chamfer_l2"] += metrics.chamfer_l2
            accumulated["metric_steps"] += 1
            if metrics.udf_smooth_l1 is not None:
                accumulated["udf_smooth_l1"] += metrics.udf_smooth_l1
                udf_metric_steps += 1
            code_indices.extend(outputs.encoding_indices.detach().cpu().tolist())

    if accumulated["metric_steps"] == 0:
        raise ValueError("Evaluation dataset is empty.")

    step_count = float(accumulated["metric_steps"])
    reconstruction = {
        "xyz_mse": accumulated["xyz_mse"] / step_count,
        "rgb_mse": accumulated["rgb_mse"] / step_count,
        "chamfer_l2": accumulated["chamfer_l2"] / step_count,
        "udf_smooth_l1": (accumulated["udf_smooth_l1"] / float(udf_metric_steps)) if udf_metric_steps > 0 else None,
    }
    codebook = asdict(summarize_codebook_usage(code_indices, codebook_size=int(model.config.codebook_size)))
    compression = asdict(summarize_compression(sample_lengths, sample_unit=args.sample_unit))
    dataset_info = {
        "ply_file_count": len(eval_paths),
        "sample_count": len(dataset),
        "sample_unit": args.sample_unit,
    }
    report = EvaluationReport(
        config={
            "checkpoint": args.checkpoint,
            "device": args.device,
            "batch_size": args.batch_size,
            "sample_unit": args.sample_unit,
        },
        dataset=dataset_info,
        reconstruction=reconstruction,
        codebook=codebook,
        compression=compression,
        notes=[
            "当前评估重点覆盖重建误差、码本利用率和压缩率。",
            "稳定性、检索与下游 probe 指标仍需后续补充。",
            "node 模式下的 UDF 指标基于 partial 点云最近邻距离构造的 observed UDF baseline。",
        ],
    )
    artifacts = write_evaluation_bundle(report, args.out)
    print(f"Wrote evaluation metrics to: {artifacts['json']}")
    print(f"Wrote evaluation summary to: {artifacts['markdown']}")
    print(f"Wrote evaluation csv to: {artifacts['csv']}")


def _parse_semantic_ids(raw: str) -> set[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return {int(value) for value in values}


if __name__ == "__main__":
    main()
