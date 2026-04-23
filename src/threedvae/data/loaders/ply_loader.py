from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from threedvae.data.schema import PlyFrameData


_PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}

_REQUIRED_FIELDS = ("x", "y", "z", "red", "green", "blue", "instance", "semantic")


def load_ply_frame(path: str, scene_id: str | None = None, frame_id: str | None = None) -> PlyFrameData:
    ply_path = Path(path)
    with ply_path.open("rb") as handle:
        fmt, vertex_count, field_names, field_types, data_start = _parse_header(handle)
        _validate_fields(field_names)

        if fmt == "ascii":
            raw = _read_ascii_vertices(handle, vertex_count, field_names, field_types)
        elif fmt == "binary_little_endian":
            raw = _read_binary_vertices(handle, vertex_count, field_names, field_types, little_endian=True)
        elif fmt == "binary_big_endian":
            raw = _read_binary_vertices(handle, vertex_count, field_names, field_types, little_endian=False)
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32, copy=False)
    rgb = np.stack([raw["red"], raw["green"], raw["blue"]], axis=1).astype(np.uint8, copy=False)
    instance_id = raw["instance"].astype(np.int32, copy=False)
    semantic_id = raw["semantic"].astype(np.int32, copy=False)
    del data_start  # Header parsing keeps this for debugging symmetry.

    return PlyFrameData(
        scene_id=scene_id or ply_path.parent.name or "default_scene",
        frame_id=frame_id or ply_path.stem,
        ply_path=str(ply_path),
        xyz=xyz,
        rgb=rgb,
        instance_id=instance_id,
        semantic_id=semantic_id,
    )


def _parse_header(handle) -> tuple[str, int, list[str], list[str], int]:
    first_line = handle.readline().decode("ascii", errors="strict").strip()
    if first_line != "ply":
        raise ValueError("File is not a PLY file.")

    fmt: str | None = None
    vertex_count = 0
    field_names: list[str] = []
    field_types: list[str] = []
    in_vertex_element = False

    while True:
        line = handle.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header.")

        decoded = line.decode("ascii", errors="strict").strip()
        if decoded == "end_header":
            break

        if not decoded or decoded.startswith("comment"):
            continue

        parts = decoded.split()
        keyword = parts[0]

        if keyword == "format":
            fmt = parts[1]
        elif keyword == "element":
            in_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex_element:
                vertex_count = int(parts[2])
        elif keyword == "property" and in_vertex_element:
            if parts[1] == "list":
                raise ValueError("Input vertex element does not support list properties.")
            field_types.append(parts[1])
            field_names.append(parts[2])

    if fmt is None:
        raise ValueError("PLY header is missing format declaration.")

    return fmt, vertex_count, field_names, field_types, handle.tell()


def _validate_fields(field_names: list[str]) -> None:
    missing = [name for name in _REQUIRED_FIELDS if name not in field_names]
    if missing:
        raise ValueError(f"PLY file is missing required fields: {missing}")


def _read_ascii_vertices(handle, vertex_count: int, field_names: list[str], field_types: list[str]) -> dict[str, np.ndarray]:
    field_arrays: dict[str, list[float]] = {name: [] for name in field_names}

    for _ in range(vertex_count):
        parts = handle.readline().decode("ascii", errors="strict").strip().split()
        if len(parts) != len(field_names):
            raise ValueError("ASCII PLY vertex row length does not match header.")

        for value, name, field_type in zip(parts, field_names, field_types):
            caster = float if field_type in {"float", "double"} else int
            field_arrays[name].append(caster(value))

    return {
        name: np.asarray(values, dtype=np.dtype(_PLY_TYPE_TO_DTYPE[field_type]))
        for name, values, field_type in zip(field_names, field_arrays.values(), field_types)
    }


def _read_binary_vertices(
    handle,
    vertex_count: int,
    field_names: list[str],
    field_types: list[str],
    *,
    little_endian: bool,
) -> dict[str, np.ndarray]:
    byte_order = "<" if little_endian else ">"
    numpy_dtype = np.dtype([(name, byte_order + _PLY_TYPE_TO_DTYPE[field_type]) for name, field_type in zip(field_names, field_types)])
    vertex_data = handle.read(vertex_count * numpy_dtype.itemsize)
    if len(vertex_data) != vertex_count * numpy_dtype.itemsize:
        raise ValueError("Binary PLY ended before all vertex data was read.")

    structured = np.frombuffer(vertex_data, dtype=numpy_dtype, count=vertex_count)
    return {name: structured[name] for name in field_names}

