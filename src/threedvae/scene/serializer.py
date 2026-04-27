from __future__ import annotations

from threedvae.data.schema import (
    CompactInstanceHeaderTokenV2,
    CompactInstanceNodeToken,
    CompactInstanceTokenBlock,
    CompactInstanceNodeTokenV2,
    CompactInstanceTokenBlockV2,
    CompactSceneTokenPacket,
    CompactSceneTokenPacketV2,
    LinearizedSceneToken,
    PoseToken,
    SceneFrame,
    SceneTokenPacket,
    SerializedTokenSequence,
)
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.tokenizer.instance_encoder import NodeCodeProvider, encode_instance


def build_scene_token_packet(
    scene: SceneFrame,
    *,
    octree_config: OctreeBuildConfig | None = None,
    node_code_provider: NodeCodeProvider | None = None,
) -> SceneTokenPacket:
    pose_tokens = [
        PoseToken(
            instance_id=instance.instance_id,
            semantic_id=instance.semantic_id,
            center_xyz=instance.pose_ego.center_xyz.copy(),
            yaw=float(instance.pose_ego.yaw),
            box_size_xyz=instance.bbox_ego.size_xyz.copy(),
        )
        for instance in scene.instances
    ]

    instance_blocks = []
    config = octree_config or OctreeBuildConfig()
    for instance in scene.instances:
        encoded = encode_instance(instance, octree_config=config, node_code_provider=node_code_provider)
        instance_blocks.append(encoded.token_block)

    return SceneTokenPacket(
        scene_id=scene.scene_id,
        frame_id=scene.frame_id,
        pose_tokens=pose_tokens,
        instance_blocks=instance_blocks,
    )


def build_compact_scene_token_packet(
    scene: SceneFrame,
    *,
    octree_config: OctreeBuildConfig | None = None,
    node_code_provider: NodeCodeProvider | None = None,
) -> CompactSceneTokenPacket:
    full_packet = build_scene_token_packet(
        scene,
        octree_config=octree_config,
        node_code_provider=node_code_provider,
    )
    compact_blocks = [_compact_instance_block(block) for block in full_packet.instance_blocks]
    return CompactSceneTokenPacket(
        scene_id=full_packet.scene_id,
        frame_id=full_packet.frame_id,
        pose_tokens=full_packet.pose_tokens,
        instance_blocks=compact_blocks,
    )


def build_llm_token_sequence(
    scene: SceneFrame,
    *,
    octree_config: OctreeBuildConfig | None = None,
    node_code_provider: NodeCodeProvider | None = None,
) -> SerializedTokenSequence:
    compact_packet = build_compact_scene_token_packet(
        scene,
        octree_config=octree_config,
        node_code_provider=node_code_provider,
    )
    sequence_tokens: list[LinearizedSceneToken] = []
    sequence_tokens.append(
        LinearizedSceneToken(
            token_type="SCENE_START",
            payload={
                "scene_id": compact_packet.scene_id,
                "frame_id": compact_packet.frame_id,
                "num_instances": len(compact_packet.instance_blocks),
            },
        )
    )

    for pose in compact_packet.pose_tokens:
        sequence_tokens.append(
            LinearizedSceneToken(
                token_type="POSE",
                payload={
                    "instance_id": pose.instance_id,
                    "semantic_id": pose.semantic_id,
                    "center_xyz": pose.center_xyz.tolist(),
                    "yaw": float(pose.yaw),
                    "box_size_xyz": pose.box_size_xyz.tolist(),
                },
            )
        )

    for block in compact_packet.instance_blocks:
        sequence_tokens.append(
            LinearizedSceneToken(
                token_type="INSTANCE_START",
                payload={
                    "instance_id": block.instance_id,
                    "semantic_id": block.semantic_id,
                    "structure_type": block.structure_type,
                    "root_center_local": block.root_center_local.tolist(),
                    "root_size_local": block.root_size_local.tolist(),
                    "max_depth": block.max_depth,
                    "traversal_order": block.traversal_order,
                    "child_slot_order": block.child_slot_order,
                    "num_node_tokens": len(block.tokens),
                },
            )
        )
        for node in block.tokens:
            sequence_tokens.append(_linearize_compact_node(node))
        sequence_tokens.append(
            LinearizedSceneToken(
                token_type="INSTANCE_END",
                payload={
                    "instance_id": block.instance_id,
                    "semantic_id": block.semantic_id,
                },
            )
        )

    sequence_tokens.append(
        LinearizedSceneToken(
            token_type="SCENE_END",
            payload={
                "scene_id": compact_packet.scene_id,
                "frame_id": compact_packet.frame_id,
            },
        )
    )
    return SerializedTokenSequence(
        scene_id=compact_packet.scene_id,
        frame_id=compact_packet.frame_id,
        sequence_format="scene_pose_then_instance_preorder_v1",
        tokens=sequence_tokens,
    )


def build_compact_scene_token_packet_v2(
    scene: SceneFrame,
    *,
    octree_config: OctreeBuildConfig | None = None,
    node_code_provider: NodeCodeProvider | None = None,
) -> CompactSceneTokenPacketV2:
    compact_packet = build_compact_scene_token_packet(
        scene,
        octree_config=octree_config,
        node_code_provider=node_code_provider,
    )
    instance_blocks = [
        _compact_instance_block_v2(pose, block)
        for pose, block in zip(compact_packet.pose_tokens, compact_packet.instance_blocks, strict=True)
    ]
    return CompactSceneTokenPacketV2(
        scene_id=compact_packet.scene_id,
        frame_id=compact_packet.frame_id,
        pose_quantization={
            "center_xyz_step_m": 0.1,
            "yaw_step_deg": 5.0,
            "box_size_step_m": 0.1,
        },
        instance_blocks=instance_blocks,
    )


def build_llm_token_sequence_v2(
    scene: SceneFrame,
    *,
    octree_config: OctreeBuildConfig | None = None,
    node_code_provider: NodeCodeProvider | None = None,
) -> SerializedTokenSequence:
    compact_v2 = build_compact_scene_token_packet_v2(
        scene,
        octree_config=octree_config,
        node_code_provider=node_code_provider,
    )
    sequence_tokens: list[LinearizedSceneToken] = [
        LinearizedSceneToken(
            token_type="SCENE_START",
            payload={
                "num_instances": len(compact_v2.instance_blocks),
                "pose_quantization": compact_v2.pose_quantization,
            },
        )
    ]
    for block in compact_v2.instance_blocks:
        sequence_tokens.append(
            LinearizedSceneToken(
                token_type="INSTANCE_HEADER",
                payload={
                    "semantic_id": block.header.semantic_id,
                    "center_xyz_q": block.header.center_xyz_q,
                    "yaw_q": block.header.yaw_q,
                    "box_size_xyz_q": block.header.box_size_xyz_q,
                    "node_count": block.header.node_count,
                    "main_code_scheme": block.header.main_code_scheme,
                },
            )
        )
        for node in block.tokens:
            sequence_tokens.append(
                LinearizedSceneToken(
                    token_type="INSTANCE_NODE_V2",
                    payload={
                        "split_flag": node.split_flag,
                        "child_mask": node.child_mask,
                        "main_code": node.main_code,
                    },
                )
            )
    sequence_tokens.append(LinearizedSceneToken(token_type="SCENE_END", payload={}))
    return SerializedTokenSequence(
        scene_id=compact_v2.scene_id,
        frame_id=compact_v2.frame_id,
        sequence_format="instance_header_then_preorder_nodes_v2",
        tokens=sequence_tokens,
    )


def _compact_instance_block(block) -> CompactInstanceTokenBlock:
    compact_tokens = [
        CompactInstanceNodeToken(
            level=token.level,
            split_flag=token.split_flag,
            child_mask=token.child_mask,
            geom_code=token.geom_code,
            rgb_code=token.rgb_code,
            num_points=token.num_points,
            learned_code=token.learned_code,
            code_source=token.code_source,
        )
        for token in block.tokens
    ]
    return CompactInstanceTokenBlock(
        instance_id=block.instance_id,
        semantic_id=block.semantic_id,
        structure_type=block.structure_type,
        root_center_local=block.root_center_local.copy(),
        root_size_local=block.root_size_local.copy(),
        max_depth=block.max_depth,
        traversal_order="preorder_dfs",
        child_slot_order="active_axes_binary",
        tokens=compact_tokens,
    )


def _linearize_compact_node(node: CompactInstanceNodeToken) -> LinearizedSceneToken:
    return LinearizedSceneToken(
        token_type="INSTANCE_NODE",
        payload={
            "level": node.level,
            "split_flag": node.split_flag,
            "child_mask": node.child_mask,
            "geom_code": node.geom_code,
            "rgb_code": node.rgb_code,
            "learned_code": node.learned_code,
            "code_source": node.code_source,
            "num_points": node.num_points,
        },
    )


def _compact_instance_block_v2(pose: PoseToken, block: CompactInstanceTokenBlock) -> CompactInstanceTokenBlockV2:
    main_code_scheme = _infer_main_code_scheme(block.tokens)
    compact_tokens = [
        CompactInstanceNodeTokenV2(
            split_flag=token.split_flag,
            child_mask=token.child_mask,
            main_code=_main_code_from_token(token, main_code_scheme),
        )
        for token in block.tokens
    ]
    return CompactInstanceTokenBlockV2(
        header=CompactInstanceHeaderTokenV2(
            semantic_id=block.semantic_id,
            center_xyz_q=_quantize_xyz(pose.center_xyz, step=0.1),
            yaw_q=_quantize_yaw_deg(pose.yaw, step_deg=5.0),
            box_size_xyz_q=_quantize_xyz(pose.box_size_xyz, step=0.1),
            node_count=len(compact_tokens),
            main_code_scheme=main_code_scheme,
        ),
        tokens=compact_tokens,
    )


def _infer_main_code_scheme(tokens: list[CompactInstanceNodeToken]) -> str:
    if tokens and all(token.learned_code is not None for token in tokens):
        return "learned"
    return "rule_packed"


def _main_code_from_token(token: CompactInstanceNodeToken, main_code_scheme: str) -> int:
    if main_code_scheme == "learned":
        if token.learned_code is None:
            raise ValueError("Expected learned_code to be present for all tokens in learned mode.")
        return int(token.learned_code)
    return _pack_rule_codes(token.geom_code, token.rgb_code)


def _pack_rule_codes(geom_code: int | None, rgb_code: int | None) -> int:
    geom = 0 if geom_code is None else int(geom_code) & 0x1FF
    rgb = 0 if rgb_code is None else int(rgb_code) & 0xFFFF
    return (geom << 16) | rgb


def _quantize_xyz(xyz, *, step: float) -> list[int]:
    return [int(round(float(value) / step)) for value in xyz]


def _quantize_yaw_deg(yaw_rad: float, *, step_deg: float) -> int:
    yaw_deg = float(yaw_rad) * 180.0 / 3.141592653589793
    return int(round(yaw_deg / step_deg))
