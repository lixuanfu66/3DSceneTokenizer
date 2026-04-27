from __future__ import annotations

import argparse
from pathlib import Path

from train_octree_node_vae import resolve_split, write_run_manifest
from threedvae.data.dataset import build_node_dataset_from_ply_paths
from threedvae.models.octree_node_vqvae import OctreeNodeVQVAE
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.train.octree_node_trainer import OctreeNodeTrainer, OctreeNodeTrainerConfig
from threedvae.utils.torch_compat import require_torch, torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an octree node VQVAE from a VAE checkpoint.")
    parser.add_argument("--vae-checkpoint", required=True)
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
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--rgb-weight", type=float, default=0.0)
    parser.add_argument("--occ-weight", type=float, default=0.1)
    parser.add_argument(
        "--occ-target-mode",
        choices=("soft_udf", "hard"),
        default="soft_udf",
        help="Use soft UDF-derived occupancy targets by default; `hard` keeps the thresholded query_occ labels.",
    )
    parser.add_argument("--occ-soft-distance", type=float, default=0.03)
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
    payload = torch.load(args.vae_checkpoint, map_location=args.device)
    model_config = dict(payload.get("model_config") or {})
    if not model_config:
        raise ValueError("VAE checkpoint does not contain model_config.")
    model_config["codebook_size"] = args.codebook_size
    model_config["commitment_cost"] = args.commitment_cost
    model = OctreeNodeVQVAE(**model_config)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    missing_without_quantizer = [key for key in missing if not key.startswith("quantizer.")]
    unexpected_without_quantizer = [key for key in unexpected if not key.startswith("quantizer.")]
    if missing_without_quantizer:
        raise ValueError(f"Checkpoint is missing non-quantizer keys: {missing_without_quantizer}")
    if unexpected_without_quantizer:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected_without_quantizer}")

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

    trainer = OctreeNodeTrainer(
        model,
        OctreeNodeTrainerConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            kl_weight=0.0,
            rgb_weight=args.rgb_weight,
            occ_weight=args.occ_weight,
            vq_weight=args.vq_weight,
            occ_target_mode=args.occ_target_mode,
            occ_soft_distance=args.occ_soft_distance,
            num_workers=args.num_workers,
            checkpoint_every=args.checkpoint_every,
        ),
    )
    history = trainer.fit(train_dataset, str(output_dir), val_dataset=val_dataset)
    final_loss = history["val_total_loss"][-1] if "val_total_loss" in history else history["total_loss"][-1]
    print(f"VQVAE training finished. Final tracked loss: {final_loss:.6f}")
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
