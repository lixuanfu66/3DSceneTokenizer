from __future__ import annotations

from dataclasses import dataclass

from threedvae.data.schema import InstanceNodeToken, InstanceTokenBlock, SceneInstance
from threedvae.octree.split_policy import OctreeBuildConfig
from threedvae.octree.tree import OctreeNode, build_instance_octree, build_octree_debug_records
from threedvae.tokenizer.learned_encoder import NodeCodeProvider
from threedvae.tokenizer.quantizer import quantize_geometry, quantize_rgb


@dataclass(slots=True)
class InstanceEncodingResult:
    token_block: InstanceTokenBlock
    octree_nodes: list[OctreeNode]


def encode_instance(
    instance: SceneInstance,
    *,
    octree_config: OctreeBuildConfig | None = None,
    node_code_provider: NodeCodeProvider | None = None,
) -> InstanceEncodingResult:
    config = octree_config or OctreeBuildConfig()
    nodes = build_instance_octree(instance, config)
    max_depth = max((node.level for node in nodes), default=0)

    tokens: list[InstanceNodeToken] = []
    for node in nodes:
        node_rgb = instance.rgb[node.point_indices]
        node_xyz_local = instance.xyz_local[node.point_indices]
        learned_code = None
        if node_code_provider is not None:
            learned_code = int(node_code_provider.encode_node(node_xyz_local, node_rgb))
        tokens.append(
            InstanceNodeToken(
                node_id=node.node_id,
                parent_id=node.parent_id,
                child_index=node.child_index,
                path_code=node.path_code,
                level=node.level,
                structure_token=1 if node.structure_state == "split" else 0,
                split_flag=node.split_flag,
                child_mask=node.child_mask,
                visibility_state=node.visibility_state,
                geom_code=quantize_geometry(node),
                rgb_code=quantize_rgb(node_rgb),
                learned_code=learned_code,
                code_source="learned_vq+rule" if learned_code is not None else "rule",
                num_points=int(node.point_indices.shape[0]),
            )
        )

    block = InstanceTokenBlock(
        instance_id=instance.instance_id,
        semantic_id=instance.semantic_id,
        structure_type="adaptive_split_tree",
        root_center_local=nodes[0].center_local.copy() if nodes else instance.bbox_ego.center_xyz.copy(),
        root_size_local=nodes[0].size_local.copy() if nodes else instance.bbox_ego.size_xyz.copy(),
        max_depth=max_depth,
        tokens=tokens,
    )
    return InstanceEncodingResult(token_block=block, octree_nodes=nodes)
