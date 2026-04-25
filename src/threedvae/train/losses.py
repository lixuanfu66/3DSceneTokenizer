from __future__ import annotations

from dataclasses import dataclass

from threedvae.eval.metrics import chamfer_l2_torch
from threedvae.utils.torch_compat import HAS_TORCH, F, require_torch, torch


@dataclass(slots=True)
class LossBreakdown:
    total_loss: object
    xyz_loss: object
    rgb_loss: object
    vq_loss: object
    udf_loss: object


def instance_tokenizer_loss(
    recon_points,
    target_xyz,
    target_rgb,
    vq_loss,
    *,
    udf_logits=None,
    target_udf=None,
    xyz_weight: float = 1.0,
    rgb_weight: float = 0.25,
    vq_weight: float = 1.0,
    udf_weight: float = 0.5,
) -> LossBreakdown:
    if not HAS_TORCH:
        require_torch()

    recon_xyz = recon_points[:, :, :3]
    recon_rgb = recon_points[:, :, 3:].clamp(0.0, 1.0)
    xyz_loss = chamfer_l2_torch(recon_xyz, target_xyz)
    rgb_loss = _nearest_neighbor_rgb_loss(
        recon_xyz=recon_xyz,
        recon_rgb=recon_rgb,
        target_xyz=target_xyz,
        target_rgb=target_rgb,
    )
    udf_loss = vq_loss * 0.0
    if udf_logits is not None and target_udf is not None:
        predicted_udf = F.softplus(udf_logits)
        udf_loss = F.smooth_l1_loss(predicted_udf, target_udf)
    total = (
        xyz_weight * xyz_loss
        + rgb_weight * rgb_loss
        + vq_weight * vq_loss
        + udf_weight * udf_loss
    )
    return LossBreakdown(
        total_loss=total,
        xyz_loss=xyz_loss,
        rgb_loss=rgb_loss,
        vq_loss=vq_loss,
        udf_loss=udf_loss,
    )


def _nearest_neighbor_rgb_loss(
    *,
    recon_xyz,
    recon_rgb,
    target_xyz,
    target_rgb,
):
    dist_sq = torch.sum((recon_xyz[:, :, None, :] - target_xyz[:, None, :, :]) ** 2, dim=-1)
    forward_indices = torch.argmin(dist_sq, dim=2)
    backward_indices = torch.argmin(dist_sq, dim=1)

    target_rgb_forward = torch.gather(
        target_rgb,
        dim=1,
        index=forward_indices[:, :, None].expand(-1, -1, target_rgb.shape[-1]),
    )
    recon_rgb_backward = torch.gather(
        recon_rgb,
        dim=1,
        index=backward_indices[:, :, None].expand(-1, -1, recon_rgb.shape[-1]),
    )

    forward_loss = F.mse_loss(recon_rgb, target_rgb_forward)
    backward_loss = F.mse_loss(recon_rgb_backward, target_rgb)
    return 0.5 * (forward_loss + backward_loss)
