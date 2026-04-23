from __future__ import annotations

import argparse
from pathlib import Path

from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.pipelines.build_scene_tokens import run_single_frame_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-frame scene tokens and debug exports.")
    parser.add_argument("--ply", required=True, help="Path to the input PLY file.")
    parser.add_argument("--out", required=True, help="Output directory for tokens and debug artifacts.")
    parser.add_argument("--scene-id", default=None, help="Optional override for scene id.")
    parser.add_argument("--frame-id", default=None, help="Optional override for frame id.")
    parser.add_argument("--near-distance", type=float, default=15.0, help="Near distance threshold in meters.")
    parser.add_argument("--mid-distance", type=float, default=35.0, help="Mid distance threshold in meters.")
    parser.add_argument("--min-points-per-node", type=int, default=8, help="Minimum points required before a node can split.")
    parser.add_argument("--geom-threshold", type=float, default=0.35, help="Geometry split threshold.")
    parser.add_argument("--rgb-threshold", type=float, default=0.10, help="RGB split threshold.")
    parser.add_argument(
        "--high-priority-semantics",
        default="",
        help="Comma-separated semantic ids for high-priority objects.",
    )
    parser.add_argument(
        "--medium-priority-semantics",
        default="",
        help="Comma-separated semantic ids for medium-priority objects.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OctreeBuildConfig(
        near_distance=args.near_distance,
        mid_distance=args.mid_distance,
        min_points_per_node=args.min_points_per_node,
        geom_threshold=args.geom_threshold,
        rgb_threshold=args.rgb_threshold,
        high_priority_semantics=_parse_semantic_ids(args.high_priority_semantics),
        medium_priority_semantics=_parse_semantic_ids(args.medium_priority_semantics),
    )
    output_path = run_single_frame_pipeline(
        ply_path=args.ply,
        output_dir=args.out,
        scene_id=args.scene_id,
        frame_id=args.frame_id,
        octree_config=config,
    )
    output_root = Path(args.out)
    print(f"Wrote debug scene token packet to: {output_path}")
    print(f"Recommended compact packet: {output_root / 'scene_tokens_compact_latest.json'}")
    print(f"Recommended LLM sequence: {output_root / 'scene_tokens_llm_sequence_latest.json'}")
    print(f"Bundle manifest: {output_root / 'scene_token_bundle.json'}")


def _parse_semantic_ids(raw: str) -> set[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return {int(value) for value in values}


if __name__ == "__main__":
    main()
