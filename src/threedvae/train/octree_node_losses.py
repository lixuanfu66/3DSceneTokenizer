from __future__ import annotations

from dataclasses import dataclass

from threedvae.utils.torch_compat import HAS_TORCH, F, require_torch, torch


@dataclass(slots=True)
class OctreeNodeLossBreakdown:
    total_loss: object
    udf_loss: object
    occ_loss: object
    kl_loss: object
    rgb_loss: object
    vq_loss: object


def octree_node_vae_loss(
    *,
    pred_udf,
    pred_occ_logits,
    target_udf,
    target_occ,
    mu,
    logvar,
    pred_rgb=None,
    target_rgb=None,
    rgb_mask=None,
    vq_loss=None,
    udf_weight: float = 1.0,
    occ_weight: float = 0.1,
    kl_weight: float = 1e-4,
    rgb_weight: float = 0.0,
    vq_weight: float = 1.0,
    occ_target_mode: str = "soft_udf",
    occ_soft_distance: float = 0.03,
    udf_loss_mode: str = "smooth_l1",
    udf_near_weight: float = 1.0,
    udf_band_weight: float = 1.0,
    udf_mid_weight: float = 1.0,
    udf_far_weight: float = 1.0,
    udf_near_threshold: float = 0.003,
    udf_band_threshold: float = 0.01,
    udf_mid_threshold: float = 0.03,
) -> OctreeNodeLossBreakdown:
    if not HAS_TORCH:
        require_torch()

    udf = _udf_loss(
        pred_udf,
        target_udf,
        mode=udf_loss_mode,
        near_weight=udf_near_weight,
        band_weight=udf_band_weight,
        mid_weight=udf_mid_weight,
        far_weight=udf_far_weight,
        near_threshold=udf_near_threshold,
        band_threshold=udf_band_threshold,
        mid_threshold=udf_mid_threshold,
    )
    occ_target = _occupancy_target_from_udf(
        target_udf=target_udf,
        fallback_target=target_occ,
        mode=occ_target_mode,
        soft_distance=occ_soft_distance,
    )
    occ = F.binary_cross_entropy_with_logits(pred_occ_logits, occ_target)
    kl = _kl_normal(mu, logvar)
    rgb = pred_udf.new_tensor(0.0)
    if pred_rgb is not None and target_rgb is not None and rgb_weight > 0.0:
        if rgb_mask is None:
            rgb = F.mse_loss(pred_rgb, target_rgb)
        else:
            rgb = _masked_mse(pred_rgb, target_rgb, rgb_mask)
    vq = pred_udf.new_tensor(0.0) if vq_loss is None else vq_loss
    total = (
        udf_weight * udf
        + occ_weight * occ
        + kl_weight * kl
        + rgb_weight * rgb
        + vq_weight * vq
    )
    return OctreeNodeLossBreakdown(
        total_loss=total,
        udf_loss=udf,
        occ_loss=occ,
        kl_loss=kl,
        rgb_loss=rgb,
        vq_loss=vq,
    )


def _kl_normal(mu, logvar):
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def _udf_loss(
    pred,
    target,
    *,
    mode: str,
    near_weight: float,
    band_weight: float,
    mid_weight: float,
    far_weight: float,
    near_threshold: float,
    band_threshold: float,
    mid_threshold: float,
):
    normalized_mode = mode.strip().lower()
    if normalized_mode == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if normalized_mode != "bucket_weighted_smooth_l1":
        raise ValueError(f"Unsupported UDF loss mode: {mode}")
    raw = F.smooth_l1_loss(pred, target, reduction="none")
    weights = torch.full_like(target, float(far_weight))
    weights = torch.where(target < float(mid_threshold), torch.full_like(weights, float(mid_weight)), weights)
    weights = torch.where(target < float(band_threshold), torch.full_like(weights, float(band_weight)), weights)
    weights = torch.where(target < float(near_threshold), torch.full_like(weights, float(near_weight)), weights)
    denom = torch.clamp(weights.sum(), min=1.0)
    return (raw * weights).sum() / denom


def _occupancy_target_from_udf(*, target_udf, fallback_target, mode: str, soft_distance: float):
    normalized_mode = mode.strip().lower()
    if normalized_mode == "hard":
        return fallback_target
    if normalized_mode != "soft_udf":
        raise ValueError(f"Unsupported occupancy target mode: {mode}")
    distance = max(float(soft_distance), 1e-6)
    return torch.exp(-torch.clamp(target_udf, min=0.0) / distance)


def _masked_mse(pred, target, mask):
    mask = mask.to(dtype=pred.dtype)
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    squared = (pred - target).pow(2) * mask
    denom = torch.clamp(mask.sum() * pred.shape[-1], min=1.0)
    return squared.sum() / denom
