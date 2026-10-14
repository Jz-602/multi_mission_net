from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _ParseHeaderLine(line: bytes) -> Tuple[str, List[str]]:
    text = line.decode("utf-8", errors="ignore").strip()
    if text == "" or text.startswith("#"):
        return "", []
    parts = text.split()
    return parts[0].upper(), parts[1:]


def _ReadPcdHeader(pcd_path: Path) -> Tuple[Dict[str, List[str]], int]:
    header: Dict[str, List[str]] = {}
    offset = 0

    with pcd_path.open("rb") as file_object:
        while True:
            line = file_object.readline()
            if line == b"":
                break
            offset += len(line)
            key, values = _ParseHeaderLine(line)
            if key == "":
                continue
            header[key] = values
            if key == "DATA":
                return header, offset

    raise ValueError(f"PCD header missing DATA line: {pcd_path}")


def _BuildFieldLayout(
    header: Dict[str, List[str]],
    pcd_path: Path,
) -> Tuple[List[str], List[int], List[int], List[str]]:
    fields = header.get("FIELDS")
    if not fields:
        raise ValueError(f"PCD header missing FIELDS: {pcd_path}")

    field_count = len(fields)
    sizes = [int(value) for value in header.get("SIZE", ["4"] * field_count)]
    types = header.get("TYPE", ["F"] * field_count)
    counts = [int(value) for value in header.get("COUNT", ["1"] * field_count)]

    if len(sizes) != field_count or len(types) != field_count or len(counts) != field_count:
        raise ValueError(f"PCD header field metadata length mismatch: {pcd_path}")

    return fields, sizes, counts, types


def _PointCount(header: Dict[str, List[str]], pcd_path: Path) -> int:
    if "POINTS" in header:
        return int(header["POINTS"][0])
    if "WIDTH" in header and "HEIGHT" in header:
        return int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
    raise ValueError(f"PCD header missing POINTS or WIDTH/HEIGHT: {pcd_path}")


def _ExpandedAsciiFieldNames(fields: List[str], counts: List[int]) -> List[str]:
    expanded_fields: List[str] = []
    for field_name, count in zip(fields, counts):
        if count == 1:
            expanded_fields.append(field_name)
        else:
            for count_idx in range(count):
                expanded_fields.append(f"{field_name}_{count_idx}")
    return expanded_fields


def _ReadAsciiXYZ(data_offset: int, fields: List[str], counts: List[int], pcd_path: Path) -> np.ndarray:
    with pcd_path.open("rb") as file_object:
        file_object.seek(data_offset)
        data = np.loadtxt(file_object, dtype=np.float32, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    expanded_fields = _ExpandedAsciiFieldNames(fields, counts)
    try:
        xyz_indices = [expanded_fields.index(axis_name) for axis_name in ("x", "y", "z")]
    except ValueError as exc:
        raise ValueError(f"PCD FIELDS must contain x y z: {pcd_path}") from exc

    return data[:, xyz_indices].astype(np.float32, copy=False)


def _ReadBinaryXYZ(
    data_offset: int,
    point_count: int,
    fields: List[str],
    sizes: List[int],
    counts: List[int],
    types: List[str],
    pcd_path: Path,
) -> np.ndarray:
    try:
        xyz_indices = [fields.index(axis_name) for axis_name in ("x", "y", "z")]
    except ValueError as exc:
        raise ValueError(f"PCD FIELDS must contain x y z: {pcd_path}") from exc

    if any(count != 1 for count in counts):
        raise ValueError(f"binary PCD COUNT must be 1 for direct xyz reading: {pcd_path}")
    if any(type_name.upper() != "F" for type_name in types):
        raise ValueError(f"binary PCD fields must be float for direct xyz reading: {pcd_path}")
    if len(set(sizes)) != 1 or sizes[0] not in (4, 8):
        raise ValueError(f"binary PCD float fields must all use SIZE 4 or SIZE 8: {pcd_path}")

    field_count = len(fields)
    dtype = np.dtype("<f4" if sizes[0] == 4 else "<f8")
    with pcd_path.open("rb") as file_object:
        file_object.seek(data_offset)
        point_array = np.fromfile(file_object, dtype=dtype, count=point_count * field_count)
    if point_array.size != point_count * field_count:
        raise ValueError(
            f"PCD binary payload too short: expected={point_count * field_count}, "
            f"actual={point_array.size}, path={pcd_path}"
        )

    point_array = point_array.reshape(point_count, field_count)
    if xyz_indices == [0, 1, 2]:
        points_xyz_lidar = point_array[:, :3]
    else:
        points_xyz_lidar = point_array[:, xyz_indices]
    return points_xyz_lidar.astype(np.float32, copy=False)


def ReadPcdXYZ(pcd_path) -> np.ndarray:
    """
    读取 PCD 点云，只返回 xyz。

    输入 pcd_path: PCD 文件路径，要求字段中包含 x/y/z。点云坐标系保持文件原始坐标系；
    在 Waymo 数据流中为当前帧 LiDAR/车体坐标系，单位 m。
    输出 points_xyz_lidar: [N, 3] float32，列为 x/y/z。
    """
    pcd_path = Path(pcd_path)
    header, data_offset = _ReadPcdHeader(pcd_path)
    fields, sizes, counts, types = _BuildFieldLayout(header, pcd_path)
    point_count = _PointCount(header, pcd_path)
    data_type = header.get("DATA", [""])[0].lower()

    if data_type == "binary":
        return _ReadBinaryXYZ(
            data_offset,
            point_count,
            fields,
            sizes,
            counts,
            types,
            pcd_path,
        )
    if data_type == "ascii":
        return _ReadAsciiXYZ(data_offset, fields, counts, pcd_path)
    raise ValueError(f"unsupported PCD DATA={data_type}; only ascii/binary are supported: {pcd_path}")


def ReadCleanPcdXYZ(pcd_path) -> np.ndarray:
    """
    读取 PCD 点云并过滤非有限点。

    输出 points_xyz_lidar: [M, 3] float32，当前帧 LiDAR/车体坐标系 xyz，单位 m。
    """
    points_xyz_lidar = ReadPcdXYZ(pcd_path)
    b_finite_mask = np.isfinite(points_xyz_lidar).all(axis=1)
    return points_xyz_lidar[b_finite_mask]
