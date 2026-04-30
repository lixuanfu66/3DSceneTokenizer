from __future__ import annotations

from dataclasses import dataclass

from threedvae.models.octree_node_vae import OctreeNodeVAE, OctreeNodeVAEOutput
from threedvae.utils.torch_compat import HAS_TORCH, F, nn, require_torch, torch


@dataclass(slots=True)
class OctreeNodeVQVAEOutput(OctreeNodeVAEOutput):
    encoding_indices: object | None = None
    vq_loss: object | None = None


if HAS_TORCH:

    class VectorQuantizer(nn.Module):
        def __init__(self, codebook_size: int, embedding_dim: int, commitment_cost: float = 0.25) -> None:
            super().__init__()
            self.embedding_dim = int(embedding_dim)
            self.codebook_size = int(codebook_size)
            self.commitment_cost = float(commitment_cost)
            self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)
            self.codebook.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

        def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            flat_z = z.reshape(-1, self.embedding_dim)
            distances = (
                flat_z.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * flat_z @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(dim=1)
            )
            encoding_indices = torch.argmin(distances, dim=1)
            quantized = self.codebook(encoding_indices).view_as(z)

            codebook_loss = F.mse_loss(quantized, z.detach())
            commitment_loss = F.mse_loss(quantized.detach(), z)
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss

            quantized = z + (quantized - z).detach()
            return quantized, encoding_indices.view(z.shape[0]), vq_loss


    class EMAVectorQuantizer(nn.Module):
        def __init__(
            self,
            codebook_size: int,
            embedding_dim: int,
            commitment_cost: float = 0.25,
            decay: float = 0.99,
            epsilon: float = 1e-5,
        ) -> None:
            super().__init__()
            self.embedding_dim = int(embedding_dim)
            self.codebook_size = int(codebook_size)
            self.commitment_cost = float(commitment_cost)
            self.decay = float(decay)
            self.epsilon = float(epsilon)
            self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)
            self.codebook.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
            self.codebook.weight.requires_grad_(False)
            self.register_buffer("ema_cluster_size", torch.ones(self.codebook_size))
            self.register_buffer("ema_code_sum", self.codebook.weight.data.clone())

        def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            flat_z = z.reshape(-1, self.embedding_dim)
            distances = (
                flat_z.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * flat_z @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(dim=1)
            )
            encoding_indices = torch.argmin(distances, dim=1)
            quantized = self.codebook(encoding_indices).view_as(z)

            if self.training:
                encodings = F.one_hot(encoding_indices, self.codebook_size).to(dtype=flat_z.dtype)
                cluster_size = encodings.sum(dim=0)
                code_sum = encodings.t() @ flat_z.detach()
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
                self.ema_code_sum.mul_(self.decay).add_(code_sum, alpha=1.0 - self.decay)

                n = self.ema_cluster_size.sum()
                smoothed_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.codebook_size * self.epsilon)
                    * n
                )
                normalized_codebook = self.ema_code_sum / smoothed_size.unsqueeze(1).clamp_min(self.epsilon)
                self.codebook.weight.data.copy_(normalized_codebook)
                quantized = self.codebook(encoding_indices).view_as(z)

            vq_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
            quantized = z + (quantized - z).detach()
            return quantized, encoding_indices.view(z.shape[0]), vq_loss


    class OctreeNodeVQVAE(OctreeNodeVAE):
        def __init__(
            self,
            *,
            codebook_size: int = 4096,
            commitment_cost: float = 0.25,
            quantizer_type: str = "standard",
            ema_decay: float = 0.99,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.codebook_size = int(codebook_size)
            self.commitment_cost = float(commitment_cost)
            self.quantizer_type = quantizer_type.strip().lower()
            self.ema_decay = float(ema_decay)
            if self.quantizer_type == "standard":
                self.quantizer = VectorQuantizer(
                    codebook_size=self.codebook_size,
                    embedding_dim=self.config.latent_dim,
                    commitment_cost=self.commitment_cost,
                )
            elif self.quantizer_type == "ema":
                self.quantizer = EMAVectorQuantizer(
                    codebook_size=self.codebook_size,
                    embedding_dim=self.config.latent_dim,
                    commitment_cost=self.commitment_cost,
                    decay=self.ema_decay,
                )
            else:
                raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")

        def forward(
            self,
            *,
            xyz: torch.Tensor,
            query_xyz: torch.Tensor,
            node_center: torch.Tensor,
            node_size: torch.Tensor,
            level: torch.Tensor,
            split_flag: torch.Tensor,
            child_index: torch.Tensor,
            semantic_id: torch.Tensor,
            rgb: torch.Tensor | None = None,
            parent_latent: torch.Tensor | None = None,
            rgb_query_xyz: torch.Tensor | None = None,
        ) -> OctreeNodeVQVAEOutput:
            mu, logvar, condition = self.encode(
                xyz=xyz,
                rgb=rgb,
                node_center=node_center,
                node_size=node_size,
                level=level,
                split_flag=split_flag,
                child_index=child_index,
                semantic_id=semantic_id,
            )
            continuous_residual = mu
            quantized_residual, encoding_indices, vq_loss = self.quantizer(continuous_residual)
            latent = quantized_residual if parent_latent is None else parent_latent + quantized_residual
            udf, occ_logits, decoder_features, rgb_pred = self.decode(
                latent=latent,
                query_xyz=query_xyz,
                node_center=node_center,
                node_size=node_size,
                condition=condition,
            )
            if rgb_query_xyz is not None and self.rgb_head is not None:
                _, _, _, rgb_pred = self.decode(
                    latent=latent,
                    query_xyz=rgb_query_xyz,
                    node_center=node_center,
                    node_size=node_size,
                    condition=condition,
                )
            return OctreeNodeVQVAEOutput(
                latent=latent,
                residual_latent=quantized_residual,
                mu=mu,
                logvar=logvar,
                udf=udf,
                occ_logits=occ_logits,
                decoder_features=decoder_features,
                rgb=rgb_pred,
                encoding_indices=encoding_indices,
                vq_loss=vq_loss,
            )

        def encode_code_indices(
            self,
            *,
            xyz: torch.Tensor,
            node_center: torch.Tensor,
            node_size: torch.Tensor,
            level: torch.Tensor,
            split_flag: torch.Tensor,
            child_index: torch.Tensor,
            semantic_id: torch.Tensor,
            rgb: torch.Tensor | None = None,
        ) -> torch.Tensor:
            mu, _, _ = self.encode(
                xyz=xyz,
                rgb=rgb,
                node_center=node_center,
                node_size=node_size,
                level=level,
                split_flag=split_flag,
                child_index=child_index,
                semantic_id=semantic_id,
            )
            _, encoding_indices, _ = self.quantizer(mu)
            return encoding_indices

        def export_config(self) -> dict[str, int | float | bool]:
            config = super().export_config()
            config.update(
                {
                    "codebook_size": self.codebook_size,
                    "commitment_cost": self.commitment_cost,
                    "quantizer_type": self.quantizer_type,
                    "ema_decay": self.ema_decay,
                }
            )
            return config

else:

    class OctreeNodeVQVAE:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            require_torch()
