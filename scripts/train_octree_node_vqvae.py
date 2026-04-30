from __future__ import annotations

import argparse
from pathlib import Path

from train_octree_node_vae import resolve_split, write_run_manifest
from threedvae.data.dataset import build_node_dataset_from_ply_paths
from threedvae.models.octree_node_vqvae import OctreeNodeVQVAE
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.train.octree_node_trainer import OctreeNodeTrainer, OctreeNodeTrainerConfig
from threedvae.utils.torch_compat import DataLoader, require_torch, torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an octree node VQVAE from a VAE checkpoint.")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--ply-dir", default=None)
    parser.add_argument("--train-ply-dir", default=None)
    parser.add_argument("--val-ply-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--points-per-node", type=int, default=128)
    parser.add_argument("--queries-per-node", type=int, default=128)
    parser.add_argument(
        "--query-strategy",
        choices=("uniform", "layered", "calibrated_surface"),
        default="uniform",
        help="Sampling strategy for UDF supervision queries.",
    )
    parser.add_argument("--udf-truncation", type=float, default=0.25)
    parser.add_argument("--near-surface-threshold", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--codebook-size", type=int, default=4096)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--quantizer-type", choices=("standard", "ema"), default="standard")
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument(
        "--init-codebook-from-data",
        action="store_true",
        help="Initialize the VQ codebook from pretrained encoder latents sampled from the training set.",
    )
    parser.add_argument("--codebook-init-max-samples", type=int, default=65536)
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
    parser.add_argument(
        "--udf-loss-mode",
        choices=("smooth_l1", "bucket_weighted_smooth_l1"),
        default="smooth_l1",
        help="UDF regression loss. Weighted mode emphasizes near-surface query bands.",
    )
    parser.add_argument("--udf-near-weight", type=float, default=1.0)
    parser.add_argument("--udf-band-weight", type=float, default=1.0)
    parser.add_argument("--udf-mid-weight", type=float, default=1.0)
    parser.add_argument("--udf-far-weight", type=float, default=1.0)
    parser.add_argument("--udf-near-threshold", type=float, default=0.003)
    parser.add_argument("--udf-band-threshold", type=float, default=0.01)
    parser.add_argument("--udf-mid-threshold", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--include-leaf-only", action="store_true")
    parser.add_argument("--min-node-points", type=int, default=1)
    parser.add_argument("--octree-preset", choices=("carla", "object"), default="carla")
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
    model_config["quantizer_type"] = args.quantizer_type
    model_config["ema_decay"] = args.ema_decay
    model = OctreeNodeVQVAE(**model_config)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    missing_without_quantizer = [key for key in missing if not key.startswith("quantizer.")]
    unexpected_without_quantizer = [key for key in unexpected if not key.startswith("quantizer.")]
    if missing_without_quantizer:
        raise ValueError(f"Checkpoint is missing non-quantizer keys: {missing_without_quantizer}")
    if unexpected_without_quantizer:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected_without_quantizer}")

    train_paths, val_paths = resolve_split(args)
    octree_config = build_octree_config(args.octree_preset)
    train_dataset = build_node_dataset_from_ply_paths(
        train_paths,
        points_per_node=args.points_per_node,
        queries_per_node=args.queries_per_node,
        udf_truncation=args.udf_truncation,
        near_surface_threshold=args.near_surface_threshold,
        seed=args.seed,
        octree_config=octree_config,
        include_leaf_only=args.include_leaf_only,
        min_node_points=args.min_node_points,
        query_strategy=args.query_strategy,
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
            min_node_points=args.min_node_points,
            query_strategy=args.query_strategy,
        )
        if val_paths
        else None
    )
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(output_dir, args, train_paths, val_paths, len(train_dataset), len(val_dataset) if val_dataset else 0)
    if args.init_codebook_from_data:
        initialize_codebook_from_dataset(
            model,
            train_dataset,
            device=args.device,
            batch_size=args.batch_size,
            max_samples=args.codebook_init_max_samples,
            seed=args.seed,
        )

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
            udf_loss_mode=args.udf_loss_mode,
            udf_near_weight=args.udf_near_weight,
            udf_band_weight=args.udf_band_weight,
            udf_mid_weight=args.udf_mid_weight,
            udf_far_weight=args.udf_far_weight,
            udf_near_threshold=args.udf_near_threshold,
            udf_band_threshold=args.udf_band_threshold,
            udf_mid_threshold=args.udf_mid_threshold,
            num_workers=args.num_workers,
            checkpoint_every=args.checkpoint_every,
        ),
    )
    history = trainer.fit(train_dataset, str(output_dir), val_dataset=val_dataset)
    final_loss = history["val_total_loss"][-1] if "val_total_loss" in history else history["total_loss"][-1]
    print(f"VQVAE training finished. Final tracked loss: {final_loss:.6f}")
    print(f"Artifacts written to: {output_dir}")


def build_octree_config(preset: str) -> OctreeBuildConfig:
    if preset == "object":
        return OctreeBuildConfig.with_default_object_semantics()
    return OctreeBuildConfig.with_default_carla_semantics()


def initialize_codebook_from_dataset(
    model: OctreeNodeVQVAE,
    dataset,
    *,
    device: str,
    batch_size: int,
    max_samples: int,
    seed: int,
) -> None:
    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.torch_collate,
    )
    latents = []
    collected = 0
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            mu, _, _ = model.encode(
                xyz=moved["xyz"],
                rgb=moved["rgb"],
                node_center=moved["node_center_local"],
                node_size=moved["node_size_local"],
                level=moved["level"],
                split_flag=moved["split_flag"],
                child_index=moved["child_index"],
                semantic_id=moved["semantic_id"],
            )
            latents.append(mu.detach().cpu())
            collected += int(mu.shape[0])
            if collected >= int(max_samples):
                break
    if not latents:
        raise ValueError("Cannot initialize codebook from an empty training dataset.")
    all_latents = torch.cat(latents, dim=0)[: int(max_samples)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if all_latents.shape[0] >= model.codebook_size:
        indices = torch.randperm(all_latents.shape[0], generator=generator)[: model.codebook_size]
        codebook = all_latents[indices]
    else:
        repeats = (model.codebook_size + all_latents.shape[0] - 1) // all_latents.shape[0]
        codebook = all_latents.repeat((repeats, 1))[: model.codebook_size]
        codebook = codebook + 1e-4 * torch.randn(codebook.shape, generator=generator)
    codebook = codebook.to(model.quantizer.codebook.weight.device)
    model.quantizer.codebook.weight.data.copy_(codebook)
    if hasattr(model.quantizer, "ema_code_sum"):
        model.quantizer.ema_code_sum.data.copy_(codebook)
    if hasattr(model.quantizer, "ema_cluster_size"):
        model.quantizer.ema_cluster_size.data.fill_(1.0)


if __name__ == "__main__":
    main()
