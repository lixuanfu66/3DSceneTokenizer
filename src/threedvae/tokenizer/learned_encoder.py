from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from threedvae.data.dataset import sample_point_cloud
from threedvae.models.instance_tokenizer import PointNetVQTokenizer
from threedvae.utils.torch_compat import HAS_TORCH, require_torch, torch


class NodeCodeProvider(Protocol):
    def encode_node(self, xyz_local: np.ndarray, rgb: np.ndarray) -> int:
        ...


@dataclass(slots=True)
class LearnedNodeCodeEncoderConfig:
    checkpoint_path: str
    device: str = "cpu"
    seed: int = 0


if HAS_TORCH:

    class LearnedNodeCodeEncoder:
        def __init__(
            self,
            model: PointNetVQTokenizer,
            *,
            device: str = "cpu",
            seed: int = 0,
        ) -> None:
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
        ) -> "LearnedNodeCodeEncoder":
            payload = torch.load(checkpoint_path, map_location=device)
            model_config = payload.get("model_config")
            if not model_config:
                raise ValueError("Checkpoint does not contain model_config; please retrain with the updated trainer.")

            model = PointNetVQTokenizer(**model_config)
            model.load_state_dict(payload["model_state_dict"])
            return cls(model, device=device, seed=seed)

        @property
        def num_points(self) -> int:
            return int(self.model.num_points)

        @torch.no_grad()
        def encode_node(self, xyz_local: np.ndarray, rgb: np.ndarray) -> int:
            sampled_xyz, sampled_rgb = sample_point_cloud(
                xyz_local.astype(np.float32, copy=False),
                rgb.astype(np.uint8, copy=False),
                num_points=self.num_points,
                seed=self.seed + int(xyz_local.shape[0]),
            )
            points = np.concatenate([sampled_xyz, sampled_rgb.astype(np.float32) / 255.0], axis=1).astype(np.float32)
            tensor = torch.as_tensor(points[None, ...], dtype=torch.float32, device=self.device)
            indices = self.model.encode_code_indices(tensor)
            return int(indices[0].detach().cpu().item())

else:

    class LearnedNodeCodeEncoder:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            require_torch()

        @classmethod
        def from_checkpoint(cls, checkpoint_path: str, *, device: str = "cpu", seed: int = 0):
            del checkpoint_path, device, seed
            require_torch()
