from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from threedvae.data.dataset import (
    build_node_dataset_from_ply_dir,
    build_node_dataset_from_ply_paths,
    build_instance_dataset_from_ply_dir,
    build_instance_dataset_from_ply_paths,
)
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.models.instance_tokenizer import PointNetVQTokenizer
from threedvae.train.trainer import InstanceTokenizerTrainer, TrainerConfig
from threedvae.utils.torch_compat import require_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned instance tokenizer from PLY frames.")
    parser.add_argument("--ply-dir", default=None, help="Directory containing all PLY files. If val dir is absent, train/val will be split from this directory.")
    parser.add_argument("--train-ply-dir", default=None, help="Directory containing training PLY files.")
    parser.add_argument("--val-ply-dir", default=None, help="Optional directory containing validation PLY files.")
    parser.add_argument("--out", required=True, help="Directory to save checkpoints and history.")
    parser.add_argument(
        "--sample-unit",
        choices=("instance", "node"),
        default="node",
        help="Training unit. `node` trains on unified-tree local patches; `instance` keeps the old global-instance baseline.",
    )
    parser.add_argument("--points-per-instance", type=int, default=256, help="Fixed sampled point count per instance.")
    parser.add_argument("--points-per-node", type=int, default=128, help="Fixed sampled point count per node when using --sample-unit node.")
    parser.add_argument("--queries-per-node", type=int, default=64, help="UDF query count per node when using --sample-unit node.")
    parser.add_argument("--udf-truncation", type=float, default=0.25, help="Truncation distance for node-level UDF supervision.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden width for encoder/decoder.")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension before quantization.")
    parser.add_argument("--codebook-size", type=int, default=512, help="VQ codebook size.")
    parser.add_argument("--commitment-cost", type=float, default=0.25, help="VQ commitment cost.")
    parser.add_argument("--udf-hidden-dim", type=int, default=128, help="Hidden width for node UDF head.")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split and point sampling.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio when using --ply-dir.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save epoch checkpoint every N epochs.")
    parser.add_argument("--udf-weight", type=float, default=0.5, help="Loss weight for node UDF supervision.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm. Set <=0 to disable.")
    parser.add_argument("--include-leaf-only", action="store_true", help="When using node mode, only train on leaf nodes.")
    parser.add_argument("--high-priority-semantics", default="", help="Comma-separated semantic ids forced to XYZ/priority-aware high-detail splitting.")
    parser.add_argument("--xy-only-semantics", default="", help="Comma-separated semantic ids forced to XY split.")
    parser.add_argument("--xz-only-semantics", default="", help="Comma-separated semantic ids forced to XZ split.")
    parser.add_argument("--yz-only-semantics", default="", help="Comma-separated semantic ids forced to YZ split.")
    return parser.parse_args()


def main() -> None:
    require_torch()
    args = parse_args()
    train_paths, val_paths = resolve_split(args)
    octree_config = OctreeBuildConfig(
        high_priority_semantics=_parse_semantic_ids(args.high_priority_semantics),
        xy_only_semantics=_parse_semantic_ids(args.xy_only_semantics),
        xz_only_semantics=_parse_semantic_ids(args.xz_only_semantics),
        yz_only_semantics=_parse_semantic_ids(args.yz_only_semantics),
    )

    if args.sample_unit == "node":
        train_dataset = build_node_dataset_from_ply_paths(
            train_paths,
            points_per_node=args.points_per_node,
            queries_per_node=args.queries_per_node,
            udf_truncation=args.udf_truncation,
            seed=args.seed,
            octree_config=octree_config,
            include_leaf_only=args.include_leaf_only,
        )
        val_dataset = (
            build_node_dataset_from_ply_paths(
                val_paths,
                points_per_node=args.points_per_node,
                queries_per_node=args.queries_per_node,
                udf_truncation=args.udf_truncation,
                seed=args.seed + 10_000,
                octree_config=octree_config,
                include_leaf_only=args.include_leaf_only,
            )
            if val_paths
            else None
        )
        num_points = args.points_per_node
    else:
        train_dataset = build_instance_dataset_from_ply_paths(
            train_paths,
            points_per_instance=args.points_per_instance,
            seed=args.seed,
        )
        val_dataset = (
            build_instance_dataset_from_ply_paths(
                val_paths,
                points_per_instance=args.points_per_instance,
                seed=args.seed + 10_000,
            )
            if val_paths
            else None
        )
        num_points = args.points_per_instance

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(output_dir, args, train_paths, val_paths, len(train_dataset), len(val_dataset) if val_dataset else 0)

    model = PointNetVQTokenizer(
        num_points=num_points,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        commitment_cost=args.commitment_cost,
        udf_hidden_dim=args.udf_hidden_dim,
    )
    trainer = InstanceTokenizerTrainer(
        model,
        TrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            udf_weight=args.udf_weight,
            grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0.0 else None,
            num_workers=args.num_workers,
            checkpoint_every=args.checkpoint_every,
        ),
    )
    history = trainer.fit(train_dataset, str(output_dir), val_dataset=val_dataset)
    final_loss = history["val_total_loss"][-1] if "val_total_loss" in history else history["total_loss"][-1]
    print(f"Training finished. Final tracked loss: {final_loss:.6f}")
    print(f"Artifacts written to: {output_dir}")


def resolve_split(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.train_ply_dir:
        train_paths = list_ply_files(args.train_ply_dir)
        val_paths = list_ply_files(args.val_ply_dir) if args.val_ply_dir else []
        if not train_paths:
            raise ValueError("No training PLY files found in --train-ply-dir.")
        return train_paths, val_paths

    if not args.ply_dir:
        raise ValueError("You must provide either --ply-dir or --train-ply-dir.")

    all_paths = list_ply_files(args.ply_dir)
    if not all_paths:
        raise ValueError("No PLY files found in --ply-dir.")

    if len(all_paths) == 1 or args.val_ratio <= 0.0:
        return all_paths, []

    rng = random.Random(args.seed)
    shuffled = list(all_paths)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * args.val_ratio)))
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1
    val_paths = sorted(shuffled[:val_count])
    train_paths = sorted(shuffled[val_count:])
    return train_paths, val_paths


def list_ply_files(directory: str | None) -> list[str]:
    if not directory:
        return []
    return sorted(str(path) for path in Path(directory).glob("*.ply"))


def write_run_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    train_paths: list[str],
    val_paths: list[str],
    train_samples: int,
    val_samples: int,
) -> None:
    manifest = {
        "args": vars(args),
        "train_ply_files": train_paths,
        "val_ply_files": val_paths,
        "train_sample_count": train_samples,
        "val_sample_count": val_samples,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _parse_semantic_ids(raw: str) -> set[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return {int(value) for value in values}


if __name__ == "__main__":
    main()
