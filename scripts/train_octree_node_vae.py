from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from threedvae.data.dataset import build_node_dataset_from_ply_paths
from threedvae.models.octree_node_vae import OctreeNodeVAE
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.train.octree_node_trainer import OctreeNodeTrainer, OctreeNodeTrainerConfig
from threedvae.utils.torch_compat import require_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an octree node VAE.")
    parser.add_argument("--ply-dir", default=None)
    parser.add_argument("--train-ply-dir", default=None)
    parser.add_argument("--val-ply-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--points-per-node", type=int, default=128)
    parser.add_argument("--queries-per-node", type=int, default=128)
    parser.add_argument("--udf-truncation", type=float, default=0.25)
    parser.add_argument("--near-surface-threshold", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--semantic-vocab-size", type=int, default=256)
    parser.add_argument("--semantic-dim", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--use-rgb-fusion", action="store_true")
    parser.add_argument("--predict-rgb", action="store_true")
    parser.add_argument("--rgb-weight", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--include-leaf-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    require_torch()
    args = parse_args()
    train_paths, val_paths = resolve_split(args)
    octree_config = OctreeBuildConfig.with_default_carla_semantics()
    train_dataset = build_node_dataset_from_ply_paths(
        train_paths,
        points_per_node=args.points_per_node,
        queries_per_node=args.queries_per_node,
        udf_truncation=args.udf_truncation,
        near_surface_threshold=args.near_surface_threshold,
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
            near_surface_threshold=args.near_surface_threshold,
            seed=args.seed + 10_000,
            octree_config=octree_config,
            include_leaf_only=args.include_leaf_only,
        )
        if val_paths
        else None
    )
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(output_dir, args, train_paths, val_paths, len(train_dataset), len(val_dataset) if val_dataset else 0)

    model = OctreeNodeVAE(
        num_points=args.points_per_node,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        semantic_vocab_size=args.semantic_vocab_size,
        semantic_dim=args.semantic_dim,
        num_attention_heads=args.attention_heads,
        use_rgb_fusion=args.use_rgb_fusion,
        predict_rgb=args.predict_rgb,
    )
    trainer = OctreeNodeTrainer(
        model,
        OctreeNodeTrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            kl_weight=args.kl_weight,
            kl_warmup_ratio=args.kl_warmup_ratio,
            rgb_weight=args.rgb_weight,
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
    return sorted(shuffled[val_count:]), sorted(shuffled[:val_count])


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


if __name__ == "__main__":
    main()
