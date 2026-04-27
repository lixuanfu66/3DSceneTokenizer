from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Literal

import numpy as np


DepthEncoding = Literal["carla-rgb", "meters", "uint16-mm", "uint16-cm", "uint16-dm"]
DepthDequantization = Literal["none", "column-ramp", "column-ramp-reverse", "column-ramp-auto"]
LabelEncoding = Literal["auto", "red", "green", "blue", "gray", "rgb24"]
TargetFrame = Literal["camera", "ego", "world"]


_CAMERA_DIR_TO_SENSOR = {
    "front": "CAM_FRONT",
    "front_left": "CAM_FRONT_LEFT",
    "front_right": "CAM_FRONT_RIGHT",
    "back": "CAM_BACK",
    "back_left": "CAM_BACK_LEFT",
    "back_right": "CAM_BACK_RIGHT",
}


def infer_bench2drive_paths(route_dir: str | Path, camera: str, frame: str) -> dict[str, Path]:
    route = Path(route_dir)
    camera_name = camera.lower()
    frame_id = Path(frame).stem
    return {
        "rgb": route / "camera" / f"rgb_{camera_name}" / f"{frame_id}.jpg",
        "depth": route / "camera" / f"depth_{camera_name}" / f"{frame_id}.png",
        "semantic": route / "camera" / f"semantic_{camera_name}" / f"{frame_id}.png",
        "instance": route / "camera" / f"instance_{camera_name}" / f"{frame_id}.png",
        "annotation": route / "anno" / f"{frame_id}.json.gz",
    }


def bench2drive_sensor_key(camera: str) -> str:
    camera_name = camera.lower().replace("-", "_")
    if camera_name.startswith("cam_"):
        return camera.upper()
    if camera_name in _CAMERA_DIR_TO_SENSOR:
        return _CAMERA_DIR_TO_SENSOR[camera_name]
    return f"CAM_{camera_name.upper()}"


def load_annotation(path: str | Path) -> dict:
    anno_path = Path(path)
    opener = gzip.open if anno_path.suffix == ".gz" else open
    with opener(anno_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def camera_calibration_from_annotation(annotation: dict, camera: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    sensor_key = bench2drive_sensor_key(camera)
    sensors = annotation.get("sensors", {})
    if sensor_key not in sensors:
        available = ", ".join(sorted(str(key) for key in sensors))
        raise KeyError(f"Annotation does not contain sensor {sensor_key!r}. Available sensors: {available}")

    sensor = sensors[sensor_key]
    intrinsic = np.asarray(sensor["intrinsic"], dtype=np.float64)
    cam2ego = _optional_matrix(sensor.get("cam2ego"))
    world2cam = _optional_matrix(sensor.get("world2cam"))
    return intrinsic, cam2ego, world2cam


def load_image_array(path: str | Path) -> np.ndarray:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading JPG/PNG files requires Pillow. Install it with: pip install Pillow") from exc

    with Image.open(path) as image:
        return np.asarray(image)


def decode_depth(depth_image: np.ndarray, encoding: DepthEncoding) -> np.ndarray:
    depth = np.asarray(depth_image)
    if encoding == "meters":
        return depth.astype(np.float32, copy=False)
    if encoding == "uint16-mm":
        return depth.astype(np.float32) / 1000.0
    if encoding == "uint16-cm":
        return depth.astype(np.float32) / 100.0
    if encoding == "uint16-dm":
        return depth.astype(np.float32) / 10.0
    if encoding != "carla-rgb":
        raise ValueError(f"Unsupported depth encoding: {encoding}")
    if depth.ndim != 3 or depth.shape[2] < 3:
        raise ValueError("CARLA RGB depth decoding expects an HxWx3 image.")

    rgb = depth[..., :3].astype(np.float64)
    normalized = (rgb[..., 0] + rgb[..., 1] * 256.0 + rgb[..., 2] * 65536.0) / (256.0**3 - 1.0)
    return (normalized * 1000.0).astype(np.float32)


def adjust_depth(depth_m: np.ndarray, *, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    return depth * float(scale) + float(offset)


def dequantize_depth(
    depth_m: np.ndarray,
    mode: DepthDequantization = "none",
    *,
    semantic: np.ndarray | None = None,
    min_run: int = 2,
    semantic_aware: bool = False,
    include_semantics: set[int] | None = None,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if mode == "none":
        return depth
    if mode not in {"column-ramp", "column-ramp-reverse", "column-ramp-auto"}:
        raise ValueError(f"Unsupported depth dequantization mode: {mode}")
    if semantic_aware and semantic is not None and semantic.shape != depth.shape:
        raise ValueError(f"semantic shape {semantic.shape} does not match depth shape {depth.shape}.")
    if min_run < 2:
        min_run = 2

    result = depth.copy()
    height, width = result.shape
    for col in range(width):
        start = 0
        while start < height:
            value = result[start, col]
            semantic_value = semantic[start, col] if semantic_aware and semantic is not None else None
            include_value = (
                _is_included_semantic(semantic[start, col], include_semantics)
                if semantic is not None and include_semantics is not None
                else True
            )
            end = start + 1
            while end < height and result[end, col] == value:
                if semantic_aware and semantic is not None and semantic[end, col] != semantic_value:
                    break
                if semantic is not None and include_semantics is not None:
                    next_include = _is_included_semantic(semantic[end, col], include_semantics)
                    if next_include != include_value:
                        break
                end += 1
            run_length = end - start
            if include_value and run_length >= min_run and np.isfinite(value) and value > 0:
                # Bench2Drive's saved uint8 depth loses the sub-meter part. Spread
                # each equal-depth vertical run across one quantization bin to
                # reduce wall-like terraces in ego-frame point clouds.
                ramp = (np.arange(run_length, dtype=np.float32) + 0.5) / run_length
                if _should_reverse_depth_run(depth[:, col], start, end, mode):
                    ramp = ramp[::-1]
                result[start:end, col] = value + ramp
            start = end
    return result


def _is_included_semantic(value: np.integer | int, include_semantics: set[int] | None) -> bool:
    if include_semantics is None:
        return True
    return int(value) in include_semantics


def _should_reverse_depth_run(column: np.ndarray, start: int, end: int, mode: DepthDequantization) -> bool:
    if mode == "column-ramp-reverse":
        return True
    if mode == "column-ramp":
        return False
    if mode != "column-ramp-auto":
        return False

    above = _nearest_finite_value(column, start - 1, step=-1)
    below = _nearest_finite_value(column, end, step=1)
    if above is not None and below is not None and above != below:
        return above > below
    if above is not None and above > column[start]:
        return True
    if below is not None and below < column[start]:
        return True
    return False


def _nearest_finite_value(column: np.ndarray, idx: int, *, step: int) -> float | None:
    while 0 <= idx < column.shape[0]:
        value = float(column[idx])
        if np.isfinite(value) and value > 0:
            return value
        idx += step
    return None


def decode_label(label_image: np.ndarray, encoding: LabelEncoding) -> np.ndarray:
    label = np.asarray(label_image)
    if encoding == "auto":
        if label.ndim == 2:
            return label.astype(np.int32, copy=False)
        encoding = "red" if label.shape[2] >= 3 else "gray"
    if encoding == "gray":
        if label.ndim == 3:
            label = label[..., 0]
        return label.astype(np.int32, copy=False)
    if label.ndim != 3:
        return label.astype(np.int32, copy=False)
    if encoding == "red":
        return label[..., 0].astype(np.int32, copy=False)
    if encoding == "green":
        return label[..., 1].astype(np.int32, copy=False)
    if encoding == "blue":
        return label[..., 2].astype(np.int32, copy=False)
    if encoding == "rgb24":
        rgb = label[..., :3].astype(np.int32)
        return rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 65536
    raise ValueError(f"Unsupported label encoding: {encoding}")


def project_depth_to_points(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    *,
    stride: int = 1,
    max_depth_m: float | None = None,
    min_depth_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array.")

    sample_depth = depth[::stride, ::stride]
    v_grid, u_grid = np.indices(sample_depth.shape, dtype=np.float32)
    u = u_grid.reshape(-1) * stride
    v = v_grid.reshape(-1) * stride
    z = sample_depth.reshape(-1)

    valid = np.isfinite(z) & (z > float(min_depth_m))
    if max_depth_m is not None:
        valid &= z <= float(max_depth_m)
    u = u[valid]
    v = v[valid]
    z = z[valid]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32), np.stack([v, u], axis=1).astype(np.int64)


def transform_points(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    if transform is None:
        return points.astype(np.float32, copy=False)
    homogeneous = np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = homogeneous @ np.asarray(transform, dtype=np.float64).T
    return transformed[:, :3].astype(np.float32)


def opencv_camera_to_carla_camera(points: np.ndarray) -> np.ndarray:
    optical = np.asarray(points, dtype=np.float32)
    return np.stack([optical[:, 2], optical[:, 0], -optical[:, 1]], axis=1).astype(np.float32)


def write_point_cloud_ply(
    out_path: str | Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    instance_id: np.ndarray,
    semantic_id: np.ndarray,
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property int instance\n")
        handle.write("property int semantic\n")
        handle.write("end_header\n")
        for point, color, instance, semantic in zip(xyz, rgb, instance_id, semantic_id):
            handle.write(
                f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(instance)} {int(semantic)}\n"
            )


def filter_point_cloud_bounds(
    xyz: np.ndarray,
    rgb: np.ndarray,
    instance_id: np.ndarray,
    semantic_id: np.ndarray,
    *,
    min_x: float | None = None,
    max_x: float | None = None,
    min_y: float | None = None,
    max_y: float | None = None,
    min_z: float | None = None,
    max_z: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.ones((xyz.shape[0],), dtype=bool)
    if min_x is not None:
        mask &= xyz[:, 0] >= float(min_x)
    if max_x is not None:
        mask &= xyz[:, 0] <= float(max_x)
    if min_y is not None:
        mask &= xyz[:, 1] >= float(min_y)
    if max_y is not None:
        mask &= xyz[:, 1] <= float(max_y)
    if min_z is not None:
        mask &= xyz[:, 2] >= float(min_z)
    if max_z is not None:
        mask &= xyz[:, 2] <= float(max_z)
    return xyz[mask], rgb[mask], instance_id[mask], semantic_id[mask]


def rgbd_to_point_cloud(
    *,
    rgb_path: str | Path,
    depth_path: str | Path,
    semantic_path: str | Path,
    instance_path: str | Path,
    intrinsic: np.ndarray,
    cam2ego: np.ndarray | None = None,
    world2cam: np.ndarray | None = None,
    target_frame: TargetFrame = "ego",
    depth_encoding: DepthEncoding = "carla-rgb",
    semantic_encoding: LabelEncoding = "auto",
    instance_encoding: LabelEncoding = "rgb24",
    stride: int = 1,
    max_depth_m: float | None = None,
    depth_scale: float = 1.0,
    depth_offset: float = 0.0,
    depth_dequantization: DepthDequantization = "none",
    depth_dequantization_min_run: int = 2,
    depth_dequantization_semantic_aware: bool = False,
    depth_dequantization_include_semantics: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = _ensure_rgb(load_image_array(rgb_path))
    depth_m = adjust_depth(decode_depth(load_image_array(depth_path), depth_encoding), scale=depth_scale, offset=depth_offset)
    semantic = decode_label(load_image_array(semantic_path), semantic_encoding)
    instance = decode_label(load_image_array(instance_path), instance_encoding)
    depth_m = dequantize_depth(
        depth_m,
        depth_dequantization,
        semantic=semantic,
        min_run=depth_dequantization_min_run,
        semantic_aware=depth_dequantization_semantic_aware,
        include_semantics=depth_dequantization_include_semantics,
    )

    _validate_same_hw(rgb, depth_m, semantic, instance)
    points_camera, pixel_rc = project_depth_to_points(depth_m, intrinsic, stride=stride, max_depth_m=max_depth_m)
    rows, cols = pixel_rc[:, 0], pixel_rc[:, 1]

    points_carla_camera = opencv_camera_to_carla_camera(points_camera)

    if target_frame == "camera":
        points = points_carla_camera
    elif target_frame == "ego":
        if cam2ego is None:
            raise ValueError("target_frame='ego' requires cam2ego calibration.")
        points = transform_points(points_carla_camera, cam2ego)
    elif target_frame == "world":
        if world2cam is None:
            raise ValueError("target_frame='world' requires world2cam calibration.")
        points = transform_points(points_carla_camera, np.linalg.inv(world2cam))
    else:
        raise ValueError(f"Unsupported target frame: {target_frame}")

    return points, rgb[rows, cols], instance[rows, cols], semantic[rows, cols]


def rgbd_to_ply(
    *,
    rgb_path: str | Path,
    depth_path: str | Path,
    semantic_path: str | Path,
    instance_path: str | Path,
    out_path: str | Path,
    intrinsic: np.ndarray,
    cam2ego: np.ndarray | None = None,
    world2cam: np.ndarray | None = None,
    target_frame: TargetFrame = "ego",
    depth_encoding: DepthEncoding = "carla-rgb",
    semantic_encoding: LabelEncoding = "auto",
    instance_encoding: LabelEncoding = "rgb24",
    stride: int = 1,
    max_depth_m: float | None = None,
    depth_scale: float = 1.0,
    depth_offset: float = 0.0,
    depth_dequantization: DepthDequantization = "none",
    depth_dequantization_min_run: int = 2,
    depth_dequantization_semantic_aware: bool = False,
    depth_dequantization_include_semantics: set[int] | None = None,
) -> int:
    points, rgb, instance, semantic = rgbd_to_point_cloud(
        rgb_path=rgb_path,
        depth_path=depth_path,
        semantic_path=semantic_path,
        instance_path=instance_path,
        intrinsic=intrinsic,
        cam2ego=cam2ego,
        world2cam=world2cam,
        target_frame=target_frame,
        depth_encoding=depth_encoding,
        semantic_encoding=semantic_encoding,
        instance_encoding=instance_encoding,
        stride=stride,
        max_depth_m=max_depth_m,
        depth_scale=depth_scale,
        depth_offset=depth_offset,
        depth_dequantization=depth_dequantization,
        depth_dequantization_min_run=depth_dequantization_min_run,
        depth_dequantization_semantic_aware=depth_dequantization_semantic_aware,
        depth_dequantization_include_semantics=depth_dequantization_include_semantics,
    )
    write_point_cloud_ply(out_path, points, rgb, instance, semantic)
    return int(points.shape[0])


def _optional_matrix(value: object) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform matrix, got {matrix.shape}.")
    return matrix


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2).astype(np.uint8, copy=False)
    if arr.shape[2] < 3:
        raise ValueError(f"RGB image must have at least 3 channels, got shape {arr.shape}.")
    return arr[..., :3].astype(np.uint8, copy=False)


def _validate_same_hw(*arrays: np.ndarray) -> None:
    shapes = [np.asarray(array).shape[:2] for array in arrays]
    if len(set(shapes)) != 1:
        raise ValueError(f"RGB, depth, semantic, and instance images must share HxW, got {shapes}.")
