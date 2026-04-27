from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from threedvae.data.dataset import sample_point_cloud
from threedvae.data.schema import SceneInstance
from threedvae.models.octree_node_vqvae import OctreeNodeVQVAE
from threedvae.octree.tree import OctreeNode
from threedvae.utils.torch_compat import HAS_TORCH, require_torch, torch


@dataclass(slots=True)
class OctreeNodeCodeEncoderConfig:
    checkpoint_path: str
    device: str = "cpu"
    seed: int = 0


if HAS_TORCH:

    class OctreeNodeCodeEncoder:
        def __init__(self, model: OctreeNodeVQVAE, *, device: str = "cpu", seed: int = 0) -> None:
            self.model = model.to(device)
            self.model.eval()
            self.device = str(device)
            self.seed = int(seed)

        @classmethod
        def from_checkpoint(
            cls,
            checkpoint_path: str,
            *,
            device: str = "cpu",
            seed: int = 0,
        ) -> "OctreeNodeCodeEncoder":
            payload = torch.load(checkpoint_path, map_location=device)
            model_config = payload.get("model_config")
            if not model_config:
                raise ValueError("Checkpoint does not contain model_config.")
            model = OctreeNodeVQVAE(**model_config)
            model.load_state_dict(payload["model_state_dict"])
            return cls(model, device=device, seed=seed)

        @property
        def num_points(self) -> int:
            return int(self.model.num_points)

        @torch.no_grad()
        def encode_node(self, xyz_local: np.ndarray, rgb: np.ndarray) -> int:
            center = np.zeros((3,), dtype=np.float32)
            size = np.maximum(np.ptp(xyz_local, axis=0).astype(np.float32, copy=False), 1e-3)
            return self._encode(
                xyz_local=xyz_local,
                rgb=rgb,
                node_center=center,
                node_size=size,
                level=0,
                split_flag=0b111,
                child_index=-1,
                semantic_id=0,
            )

        @torch.no_grad()
        def encode_octree_node(
            self,
            *,
            instance: SceneInstance,
            node: OctreeNode,
            xyz_local: np.ndarray,
            rgb: np.ndarray,
        ) -> int:
            return self._encode(
                xyz_local=xyz_local,
                rgb=rgb,
                node_center=node.center_local,
                node_size=node.size_local,
                level=node.level,
                split_flag=node.split_flag,
                child_index=-1 if node.child_index is None else node.child_index,
                semantic_id=instance.semantic_id,
            )

        def _encode(
            self,
            *,
            xyz_local: np.ndarray,
            rgb: np.ndarray,
            node_center: np.ndarray,
            node_size: np.ndarray,
            level: int,
            split_flag: int,
            child_index: int,
            semantic_id: int,
        ) -> int:
            sampled_xyz, sampled_rgb = sample_point_cloud(
                xyz_local.astype(np.float32, copy=False),
                rgb.astype(np.uint8, copy=False),
                num_points=self.num_points,
                seed=self.seed + int(xyz_local.shape[0]),
            )
            xyz_tensor = torch.as_tensor(sampled_xyz[None, ...], dtype=torch.float32, device=self.device)
            rgb_tensor = torch.as_tensor(sampled_rgb[None, ...].astype(np.float32) / 255.0, dtype=torch.float32, device=self.device)
            indices = self.model.encode_code_indices(
                xyz=xyz_tensor,
                rgb=rgb_tensor,
                node_center=torch.as_tensor(node_center[None, ...], dtype=torch.float32, device=self.device),
                node_size=torch.as_tensor(node_size[None, ...], dtype=torch.float32, device=self.device),
                level=torch.as_tensor([level], dtype=torch.int64, device=self.device),
                split_flag=torch.as_tensor([split_flag], dtype=torch.int64, device=self.device),
                child_index=torch.as_tensor([child_index], dtype=torch.int64, device=self.device),
                semantic_id=torch.as_tensor([semantic_id], dtype=torch.int64, device=self.device),
            )
            return int(indices[0].detach().cpu().item())

else:

    class OctreeNodeCodeEncoder:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            require_torch()

        @classmethod
        def from_checkpoint(cls, checkpoint_path: str, *, device: str = "cpu", seed: int = 0):
            del checkpoint_path, device, seed
            require_torch()
