from __future__ import annotations

import math

import numpy as np

from threedvae.data.schema import GroundAlignedOBB, LocalFrameTransform, Pose3DYaw


def pose_from_obb(obb: GroundAlignedOBB) -> Pose3DYaw:
    return Pose3DYaw(center_xyz=obb.center_xyz.copy(), yaw=float(obb.yaw))


def local_frame_from_obb(obb: GroundAlignedOBB) -> LocalFrameTransform:
    return LocalFrameTransform(translation_xyz=obb.center_xyz.copy(), yaw=float(obb.yaw))


def transform_points_to_local(xyz_ego: np.ndarray, pose_ego: Pose3DYaw) -> np.ndarray:
    translated = xyz_ego.astype(np.float64, copy=False) - pose_ego.center_xyz.astype(np.float64, copy=False)
    cos_yaw = math.cos(-float(pose_ego.yaw))
    sin_yaw = math.sin(-float(pose_ego.yaw))
    rotation = np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (translated @ rotation.T).astype(np.float32)

