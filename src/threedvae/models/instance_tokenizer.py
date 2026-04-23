from __future__ import annotations

from dataclasses import asdict, dataclass

from threedvae.utils.torch_compat import HAS_TORCH, F, nn, require_torch, torch


@dataclass(slots=True)
class PointNetVQOutput:
    recon_points: object
    quantized_latent: object
    encoding_indices: object
    vq_loss: object
    udf_logits: object | None = None


@dataclass(slots=True)
class PointNetVQTokenizerConfig:
    num_points: int
    input_dim: int = 6
    hidden_dim: int = 128
    latent_dim: int = 64
    codebook_size: int = 512
    commitment_cost: float = 0.25
    udf_hidden_dim: int = 128


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


    class PointNetEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, points: torch.Tensor) -> torch.Tensor:
            features = self.net(points)
            return features.mean(dim=1)


    class PointCloudDecoder(nn.Module):
        def __init__(self, latent_dim: int, hidden_dim: int, num_points: int, output_dim: int) -> None:
            super().__init__()
            self.num_points = int(num_points)
            self.output_dim = int(output_dim)
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_points * self.output_dim),
            )

        def forward(self, latent: torch.Tensor) -> torch.Tensor:
            decoded = self.net(latent)
            return decoded.view(latent.shape[0], self.num_points, self.output_dim)


    class UDFDecoder(nn.Module):
        def __init__(self, latent_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim + 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, latent: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
            expanded = latent[:, None, :].expand(-1, query_xyz.shape[1], -1)
            features = torch.cat([expanded, query_xyz], dim=-1)
            return self.net(features)


    class PointNetVQTokenizer(nn.Module):
        def __init__(
            self,
            *,
            num_points: int,
            input_dim: int = 6,
            hidden_dim: int = 128,
            latent_dim: int = 64,
            codebook_size: int = 512,
            commitment_cost: float = 0.25,
            udf_hidden_dim: int = 128,
        ) -> None:
            super().__init__()
            self.config = PointNetVQTokenizerConfig(
                num_points=int(num_points),
                input_dim=int(input_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=int(latent_dim),
                codebook_size=int(codebook_size),
                commitment_cost=float(commitment_cost),
                udf_hidden_dim=int(udf_hidden_dim),
            )
            self.num_points = self.config.num_points
            self.encoder = PointNetEncoder(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                latent_dim=self.config.latent_dim,
            )
            self.pre_quant = nn.Linear(self.config.latent_dim, self.config.latent_dim)
            self.quantizer = VectorQuantizer(
                codebook_size=self.config.codebook_size,
                embedding_dim=self.config.latent_dim,
                commitment_cost=self.config.commitment_cost,
            )
            self.decoder = PointCloudDecoder(
                latent_dim=self.config.latent_dim,
                hidden_dim=self.config.hidden_dim,
                num_points=self.num_points,
                output_dim=self.config.input_dim,
            )
            self.udf_decoder = UDFDecoder(
                latent_dim=self.config.latent_dim,
                hidden_dim=self.config.udf_hidden_dim,
            )

        def forward(self, points: torch.Tensor, query_xyz: torch.Tensor | None = None) -> PointNetVQOutput:
            latent = self.encoder(points)
            latent = self.pre_quant(latent)
            quantized, encoding_indices, vq_loss = self.quantizer(latent)
            recon_points = self.decoder(quantized)
            udf_logits = None
            if query_xyz is not None:
                udf_logits = self.udf_decoder(quantized, query_xyz)
            return PointNetVQOutput(
                recon_points=recon_points,
                quantized_latent=quantized,
                encoding_indices=encoding_indices,
                vq_loss=vq_loss,
                udf_logits=udf_logits,
            )

        def encode_code_indices(self, points: torch.Tensor) -> torch.Tensor:
            latent = self.encoder(points)
            latent = self.pre_quant(latent)
            _, encoding_indices, _ = self.quantizer(latent)
            return encoding_indices

        def export_config(self) -> dict[str, int | float]:
            return asdict(self.config)

else:

    class PointNetVQTokenizer:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            require_torch()
