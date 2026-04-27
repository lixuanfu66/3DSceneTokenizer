from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from threedvae.data.loaders.ply_loader import load_ply_frame
from threedvae.debug.exporters import (
    export_instance_bbox_json,
    export_instance_octree_debug,
    export_instance_points_with_octree_node_bboxes_ply,
    export_scene_instance_bboxes_ply,
    export_scene_octree_node_bboxes_ply,
)
from threedvae.octree.split_policy import OctreeBuildConfig, build_default_carla_semantic_policies
from threedvae.octree.tree import build_octree_debug_records
from threedvae.scene.builder import build_scene_frame
from threedvae.scene.serializer import (
    build_compact_scene_token_packet,
    build_compact_scene_token_packet_v2,
    build_llm_token_sequence,
    build_llm_token_sequence_v2,
    build_scene_token_packet,
)
from threedvae.tokenizer.instance_encoder import NodeCodeProvider, encode_instance


def run_single_frame_pipeline(
    ply_path: str,
    output_dir: str,
    *,
    scene_id: str | None = None,
    frame_id: str | None = None,
    octree_config: OctreeBuildConfig | None = None,
    node_code_provider: NodeCodeProvider | None = None,
) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    config = octree_config or OctreeBuildConfig()
    semantic_name_lookup = _semantic_name_lookup(config)
    frame = load_ply_frame(ply_path, scene_id=scene_id, frame_id=frame_id)
    scene = build_scene_frame(frame, semantic_name_lookup=semantic_name_lookup)

    export_scene_instance_bboxes_ply(scene, str(output_root / "scene_instance_bboxes.ply"))

    scene_node_debug_records = []
    for instance in scene.instances:
        export_instance_bbox_json(instance, str(output_root / f"instance_{instance.instance_id}_bbox.json"))
        encoded = encode_instance(instance, octree_config=config, node_code_provider=node_code_provider)
        debug_records = build_octree_debug_records(instance, encoded.octree_nodes)
        export_instance_octree_debug(instance, debug_records, str(output_root))
        export_instance_points_with_octree_node_bboxes_ply(
            instance,
            debug_records,
            str(output_root / f"{_instance_analysis_filename(instance)}.ply"),
        )
        scene_node_debug_records.append((instance, debug_records))

    export_scene_octree_node_bboxes_ply(
        scene_node_debug_records,
        str(output_root / "scene_octree_node_bboxes.ply"),
    )

    packet = build_scene_token_packet(scene, octree_config=config, node_code_provider=node_code_provider)
    compact_packet = build_compact_scene_token_packet(scene, octree_config=config, node_code_provider=node_code_provider)
    compact_packet_v2 = build_compact_scene_token_packet_v2(scene, octree_config=config, node_code_provider=node_code_provider)
    llm_sequence = build_llm_token_sequence(scene, octree_config=config, node_code_provider=node_code_provider)
    llm_sequence_v2 = build_llm_token_sequence_v2(scene, octree_config=config, node_code_provider=node_code_provider)
    packet_path = output_root / "scene_tokens.json"
    with packet_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_to_jsonable(packet), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    compact_packet_path = output_root / "scene_tokens_compact.json"
    with compact_packet_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_to_jsonable(compact_packet), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    compact_packet_v2_path = output_root / "scene_tokens_compact_v2.json"
    with compact_packet_v2_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_to_jsonable(compact_packet_v2), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    compact_packet_latest_path = output_root / "scene_tokens_compact_latest.json"
    with compact_packet_latest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_to_jsonable(compact_packet_v2), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    llm_sequence_path = output_root / "scene_tokens_llm_sequence.json"
    with llm_sequence_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_to_jsonable(llm_sequence), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    llm_sequence_v2_path = output_root / "scene_tokens_llm_sequence_v2.json"
    with llm_sequence_v2_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_to_jsonable(llm_sequence_v2), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    llm_sequence_latest_path = output_root / "scene_tokens_llm_sequence_latest.json"
    with llm_sequence_latest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_to_jsonable(llm_sequence_v2), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    bundle_manifest = {
        "scene_id": scene.scene_id,
        "frame_id": scene.frame_id,
        "recommended_compact_packet": compact_packet_latest_path.name,
        "recommended_llm_sequence": llm_sequence_latest_path.name,
        "artifacts": {
            "debug_packet_full": packet_path.name,
            "compact_packet_v1": compact_packet_path.name,
            "compact_packet_v2": compact_packet_v2_path.name,
            "compact_packet_latest": compact_packet_latest_path.name,
            "llm_sequence_v1": llm_sequence_path.name,
            "llm_sequence_v2": llm_sequence_v2_path.name,
            "llm_sequence_latest": llm_sequence_latest_path.name,
            "scene_instance_bboxes": "scene_instance_bboxes.ply",
            "scene_octree_node_bboxes": "scene_octree_node_bboxes.ply",
        },
    }
    bundle_manifest_path = output_root / "scene_token_bundle.json"
    with bundle_manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(bundle_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return packet_path


def _to_jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {key: _to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _semantic_name_lookup(config: OctreeBuildConfig) -> dict[int, str]:
    lookup = {semantic_id: policy.tag_name for semantic_id, policy in build_default_carla_semantic_policies().items()}
    lookup.update({semantic_id: policy.tag_name for semantic_id, policy in config.semantic_policies.items()})
    return lookup


def _instance_analysis_filename(instance) -> str:
    semantic_name = _safe_filename_part(instance.category_name)
    return f"instance_{instance.instance_id}_{semantic_name}_points_and_octree_node_bboxes"


def _safe_filename_part(raw: str) -> str:
    chars = []
    for char in raw.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_"}:
            chars.append(char)
        elif char.isspace():
            chars.append("_")
    value = "".join(chars).strip("_")
    return value or "unknown"
