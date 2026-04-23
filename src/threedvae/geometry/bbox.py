from __future__ import annotations

import math

import numpy as np

from threedvae.data.schema import GroundAlignedOBB


_EPS = 1e-6


def estimate_ground_aligned_obb(xyz_ego: np.ndarray) -> GroundAlignedOBB:
    if xyz_ego.ndim != 2 or xyz_ego.shape[1] != 3:
        raise ValueError("xyz_ego must have shape [N, 3].")
    if xyz_ego.shape[0] == 0:
        raise ValueError("Cannot estimate OBB for an empty point cloud.")

    xy = xyz_ego[:, :2].astype(np.float64, copy=False)
    z = xyz_ego[:, 2].astype(np.float64, copy=False)

    yaw = _estimate_yaw_from_xy(xy)
    rotation = _rotation_2d(-yaw)
    local_xy = xy @ rotation.T

    min_xy = local_xy.min(axis=0)
    max_xy = local_xy.max(axis=0)
    min_z = float(z.min())
    max_z = float(z.max())

    center_local_xy = (min_xy + max_xy) * 0.5
    center_world_xy = center_local_xy @ _rotation_2d(yaw).T
    center_z = 0.5 * (min_z + max_z)

    size_xy = np.maximum(max_xy - min_xy, _EPS)
    size_z = max(max_z - min_z, _EPS)

    return GroundAlignedOBB(
        center_xyz=np.asarray([center_world_xy[0], center_world_xy[1], center_z], dtype=np.float32),
        size_xyz=np.asarray([size_xy[0], size_xy[1], size_z], dtype=np.float32),
        yaw=float(yaw),
    )


def obb_vertices(obb: GroundAlignedOBB) -> np.ndarray:
    half = 0.5 * obb.size_xyz.astype(np.float64, copy=False)
    local_vertices = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )

    cos_yaw = math.cos(float(obb.yaw))
    sin_yaw = math.sin(float(obb.yaw))
    rotation = np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    world_vertices = local_vertices @ rotation.T + obb.center_xyz.astype(np.float64, copy=False)
    return world_vertices.astype(np.float32)


def quad_faces() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ],
        dtype=np.int32,
    )


def _estimate_yaw_from_xy(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0

    centered = xy - xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    anisotropy = float(eigvals.max() - eigvals.min())
    if anisotropy < 1e-4:
        return 0.0
    return float(math.atan2(principal[1], principal[0]))


def _rotation_2d(yaw: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)],
        ],
        dtype=np.float64,
    )

