from __future__ import annotations

from dataclasses import dataclass
from math import log2

import numpy as np

from threedvae.utils.torch_compat import HAS_TORCH, F, require_torch, torch


@dataclass(slots=True)
class ReconstructionMetricSummary:
    xyz_mse: float
    rgb_mse: float
    udf_smooth_l1: float | None
    chamfer_l2: float


@dataclass(slots=True)
class CodebookMetricSummary:
    code_count: int
    used_code_count: int
    usage_rate: float
    entropy_bits: float
    perplexity: float


@dataclass(slots=True)
class CompressionMetricSummary:
    sample_unit: str
    sample_count: int
    avg_input_points: float
    latent_tokens_per_sample: float
    points_per_token_ratio: float


def chamfer_l2_numpy(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        raise ValueError("Chamfer distance requires non-empty point sets.")
    diff = points_a[:, None, :] - points_b[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    forward = np.mean(np.min(dist_sq, axis=1))
    backward = np.mean(np.min(dist_sq, axis=0))
    return float(forward + backward)


def summarize_codebook_usage(indices: list[int], codebook_size: int) -> CodebookMetricSummary:
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive.")
    histogram = np.zeros((codebook_size,), dtype=np.int64)
    for index in indices:
        if index < 0 or index >= codebook_size:
            raise ValueError(f"Code index {index} is out of range for codebook size {codebook_size}.")
        histogram[index] += 1

    total = int(np.sum(histogram))
    if total == 0:
        return CodebookMetricSummary(
            code_count=int(codebook_size),
            used_code_count=0,
            usage_rate=0.0,
            entropy_bits=0.0,
            perplexity=0.0,
        )

    probabilities = histogram[histogram > 0].astype(np.float64) / float(total)
    entropy_bits = float(-np.sum(probabilities * np.log2(probabilities)))
    perplexity = float(2.0 ** entropy_bits)
    used_code_count = int(probabilities.shape[0])
    return CodebookMetricSummary(
        code_count=int(codebook_size),
        used_code_count=used_code_count,
        usage_rate=float(used_code_count / float(codebook_size)),
        entropy_bits=entropy_bits,
        perplexity=perplexity,
    )


def summarize_compression(sample_lengths: list[int], *, sample_unit: str) -> CompressionMetricSummary:
    if not sample_lengths:
        return CompressionMetricSummary(
            sample_unit=sample_unit,
            sample_count=0,
            avg_input_points=0.0,
            latent_tokens_per_sample=0.0,
            points_per_token_ratio=0.0,
        )

    avg_input_points = float(np.mean(np.asarray(sample_lengths, dtype=np.float64)))
    latent_tokens_per_sample = 1.0
    return CompressionMetricSummary(
        sample_unit=sample_unit,
        sample_count=len(sample_lengths),
        avg_input_points=avg_input_points,
        latent_tokens_per_sample=latent_tokens_per_sample,
        points_per_token_ratio=float(avg_input_points / latent_tokens_per_sample),
    )


if HAS_TORCH:

    def chamfer_l2_torch(points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
        diff = points_a[:, :, None, :] - points_b[:, None, :, :]
        dist_sq = torch.sum(diff * diff, dim=-1)
        forward = torch.mean(torch.min(dist_sq, dim=2).values, dim=1)
        backward = torch.mean(torch.min(dist_sq, dim=1).values, dim=1)
        return torch.mean(forward + backward)


    @torch.no_grad()
    def batch_reconstruction_metrics(
        recon_points: torch.Tensor,
        target_xyz: torch.Tensor,
        target_rgb: torch.Tensor,
        *,
        udf_logits: torch.Tensor | None = None,
        target_udf: torch.Tensor | None = None,
    ) -> ReconstructionMetricSummary:
        recon_xyz = recon_points[:, :, :3]
        recon_rgb = recon_points[:, :, 3:].clamp(0.0, 1.0)
        xyz_mse = float(F.mse_loss(recon_xyz, target_xyz).detach().cpu())
        rgb_mse = float(F.mse_loss(recon_rgb, target_rgb).detach().cpu())
        chamfer_l2 = float(chamfer_l2_torch(recon_xyz, target_xyz).detach().cpu())
        udf_loss = None
        if udf_logits is not None and target_udf is not None:
            predicted_udf = F.softplus(udf_logits)
            udf_loss = float(F.smooth_l1_loss(predicted_udf, target_udf).detach().cpu())
        return ReconstructionMetricSummary(
            xyz_mse=xyz_mse,
            rgb_mse=rgb_mse,
            udf_smooth_l1=udf_loss,
            chamfer_l2=chamfer_l2,
        )

else:

    def chamfer_l2_torch(points_a, points_b):
        del points_a, points_b
        require_torch()


    def batch_reconstruction_metrics(recon_points, target_xyz, target_rgb, *, udf_logits=None, target_udf=None):
        del recon_points, target_xyz, target_rgb, udf_logits, target_udf
        require_torch()
