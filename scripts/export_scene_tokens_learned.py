from __future__ import annotations

import argparse

from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.pipelines.build_scene_tokens import run_single_frame_pipeline
from threedvae.tokenizer.learned_encoder import LearnedNodeCodeEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export scene tokens with a learned node code encoder.")
    parser.add_argument("--ply", required=True, help="Input single-frame PLY path.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--checkpoint", required=True, help="Trained tokenizer checkpoint path.")
    parser.add_argument("--scene-id", default=None, help="Optional scene id override.")
    parser.add_argument("--frame-id", default=None, help="Optional frame id override.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for learned node encoding.")
    parser.add_argument("--high-priority-semantics", default="", help="Comma-separated semantic ids forced to XYZ high-detail split.")
    parser.add_argument("--xy-only-semantics", default="", help="Comma-separated semantic ids forced to XY split.")
    parser.add_argument("--xz-only-semantics", default="", help="Comma-separated semantic ids forced to XZ split.")
    parser.add_argument("--yz-only-semantics", default="", help="Comma-separated semantic ids forced to YZ split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = LearnedNodeCodeEncoder.from_checkpoint(args.checkpoint, device=args.device, seed=args.seed)
    octree_config = OctreeBuildConfig(
        high_priority_semantics=_parse_semantic_ids(args.high_priority_semantics),
        xy_only_semantics=_parse_semantic_ids(args.xy_only_semantics),
        xz_only_semantics=_parse_semantic_ids(args.xz_only_semantics),
        yz_only_semantics=_parse_semantic_ids(args.yz_only_semantics),
    )
    packet_path = run_single_frame_pipeline(
        args.ply,
        args.out,
        scene_id=args.scene_id,
        frame_id=args.frame_id,
        octree_config=octree_config,
        node_code_provider=provider,
    )
    print(f"Learned scene token packet written to: {packet_path}")


def _parse_semantic_ids(raw: str) -> set[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return {int(value) for value in values}


if __name__ == "__main__":
    main()
