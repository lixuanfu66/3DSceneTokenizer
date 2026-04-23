from __future__ import annotations

import numpy as np

from threedvae.octree.tree import OctreeNode


def quantize_geometry(node: OctreeNode) -> int:
    density_bin = _bin_value(np.log1p(node.point_indices.shape[0]), 0.0, 8.0, 8)
    geom_bin = _bin_value(node.geom_score, 0.0, 1.0, 8)
    level_bin = min(node.level, 7)
    return int(density_bin * 64 + geom_bin * 8 + level_bin)


def quantize_rgb(rgb: np.ndarray) -> int:
    if rgb.shape[0] == 0:
        return 0

    mean_rgb = np.mean(rgb.astype(np.float32), axis=0) / 255.0
    std_rgb = np.mean(np.std(rgb.astype(np.float32) / 255.0, axis=0))
    r_bin = _bin_value(mean_rgb[0], 0.0, 1.0, 16)
    g_bin = _bin_value(mean_rgb[1], 0.0, 1.0, 16)
    b_bin = _bin_value(mean_rgb[2], 0.0, 1.0, 16)
    var_bin = _bin_value(std_rgb, 0.0, 0.5, 16)
    return int((r_bin << 12) | (g_bin << 8) | (b_bin << 4) | var_bin)


def _bin_value(value: float, low: float, high: float, bins: int) -> int:
    if high <= low:
        return 0
    clipped = min(max(float(value), low), high)
    if clipped == high:
        return bins - 1
    ratio = (clipped - low) / (high - low)
    return int(ratio * bins)

