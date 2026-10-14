import json
import os
import struct
from argparse import Namespace

import numpy as np

from .constants import INVALID_HEIGHT


CELL_DTYPE = np.dtype(
    [
        ("n", np.uint32),
        ("_pad", np.uint32),
        ("sum_z", np.float64),
        ("squared_sum_z", np.float64),
    ]
)


def EnsureDir(path_name):
    os.makedirs(path_name, exist_ok=True)


def ReadIniFile(filename):
    data = {}
    with open(filename, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            if line[0] in "#/":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            data[parts[0]] = parts[1]
    return data


def UpdateNamespaceFromCli(args, argv):
    args_dict = vars(args)
    for item in argv[1:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key not in args_dict:
            continue
        original_value = args_dict[key]
        if isinstance(original_value, bool):
            args_dict[key] = value.lower() in ("1", "true", "yes", "y")
        elif isinstance(original_value, int):
            args_dict[key] = int(value)
        elif isinstance(original_value, float):
            args_dict[key] = float(value)
        else:
            args_dict[key] = value
    return args


def ReadMat(filename):
    with open(filename, "rb") as fp:
        header = fp.read(12)
        rows, cols, cv_type = struct.unpack("<iii", header)
        dtype, channels = CvTypeToNumpy(cv_type)
        data = np.frombuffer(fp.read(), dtype=dtype)
    if channels > 1:
        return data.reshape(rows, cols, channels).copy()
    return data.reshape(rows, cols).copy()


def WriteMat(filename, array):
    array = np.ascontiguousarray(array)
    cv_type = NumpyToCvType(array.dtype, 1 if array.ndim == 2 else array.shape[2])
    with open(filename, "wb") as fp:
        fp.write(struct.pack("<iii", array.shape[0], array.shape[1], cv_type))
        fp.write(array.tobytes())


def CvTypeToNumpy(cv_type):
    depth = cv_type & 7
    channels = 1 + (cv_type >> 3)
    depth_map = {
        0: np.uint8,
        2: np.uint16,
        4: np.int32,
        5: np.float32,
        6: np.float64,
    }
    if depth not in depth_map:
        raise ValueError(f"Unsupported OpenCV depth: {depth}")
    return depth_map[depth], channels


def NumpyToCvType(dtype, channels):
    if dtype == np.uint8:
        depth = 0
    elif dtype == np.uint16:
        depth = 2
    elif dtype == np.int32:
        depth = 4
    elif dtype == np.float32:
        depth = 5
    elif dtype == np.float64:
        depth = 6
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")
    return depth + ((channels - 1) << 3)


def LoadPoseDb(filename):
    timestamps = []
    poses = []
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) != 7:
                continue
            timestamps.append(int(parts[0]))
            poses.append([float(item) for item in parts[1:]])
    return np.asarray(timestamps, dtype=np.int64), np.asarray(poses, dtype=np.float64)


def ReadSmallMap(filename, small_map_rows):
    with open(filename, "rb") as fp:
        center_x = struct.unpack("<i", fp.read(4))[0]
        center_y = struct.unpack("<i", fp.read(4))[0]
        cells = np.frombuffer(fp.read(), dtype=CELL_DTYPE, count=small_map_rows * small_map_rows).copy()
    return center_x, center_y, cells.reshape(small_map_rows, small_map_rows)


def LoadPcdXyzi(filename):
    with open(filename, "rb") as fp:
        lines = []
        while True:
            line = fp.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading header: {filename}")
            line_str = line.decode("utf-8").strip()
            lines.append(line_str)
            if line_str.startswith("DATA"):
                data_mode = line_str.split()[1].lower()
                break
        payload = fp.read()

    meta = ParsePcdHeader(lines)
    dtype = BuildPcdDtype(meta)
    if data_mode == "binary":
        structured = np.frombuffer(payload, dtype=dtype, count=meta["points"]).copy()
    elif data_mode == "ascii":
        flat = np.loadtxt(filename, comments=None, skiprows=len(lines))
        structured = FlatAsciiToStructured(flat, meta)
    else:
        raise ValueError(f"Unsupported PCD mode: {data_mode}")

    x = ExtractField(structured, "x")
    y = ExtractField(structured, "y")
    z = ExtractField(structured, "z")
    intensity = ExtractField(structured, "intensity", default=0.0)
    return np.column_stack([x, y, z, intensity]).astype(np.float64, copy=False)


def ParsePcdHeader(lines):
    meta = {}
    for line in lines:
        if not line:
            continue
        parts = line.split()
        key = parts[0].lower()
        values = parts[1:]
        meta[key] = values
    meta["fields"] = meta.get("fields", [])
    meta["size"] = [int(item) for item in meta.get("size", [])]
    meta["type"] = meta.get("type", [])
    meta["count"] = [int(item) for item in meta.get("count", ["1"] * len(meta["fields"]))]
    meta["points"] = int(meta.get("points", [meta.get("width", ["0"])[0]])[0])
    return meta


def BuildPcdDtype(meta):
    dtype_desc = []
    for name, size, item_type, count in zip(meta["fields"], meta["size"], meta["type"], meta["count"]):
        base_dtype = PcdScalarTypeToNumpy(item_type, size)
        if count == 1:
            dtype_desc.append((name, base_dtype))
        else:
            dtype_desc.append((name, base_dtype, (count,)))
    return np.dtype(dtype_desc)


def PcdScalarTypeToNumpy(item_type, size):
    key = (item_type.upper(), int(size))
    mapping = {
        ("F", 4): np.float32,
        ("F", 8): np.float64,
        ("U", 1): np.uint8,
        ("U", 2): np.uint16,
        ("U", 4): np.uint32,
        ("I", 1): np.int8,
        ("I", 2): np.int16,
        ("I", 4): np.int32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported PCD scalar type: {key}")
    return mapping[key]


def FlatAsciiToStructured(flat, meta):
    if flat.ndim == 1:
        flat = flat.reshape(1, -1)
    columns = []
    cursor = 0
    dtype_desc = []
    for name, size, item_type, count in zip(meta["fields"], meta["size"], meta["type"], meta["count"]):
        base_dtype = PcdScalarTypeToNumpy(item_type, size)
        if count == 1:
            dtype_desc.append((name, base_dtype))
            columns.append(flat[:, cursor].astype(base_dtype, copy=False))
        else:
            dtype_desc.append((name, base_dtype, (count,)))
            columns.append(flat[:, cursor : cursor + count].astype(base_dtype, copy=False))
        cursor += count
    structured = np.empty(flat.shape[0], dtype=np.dtype(dtype_desc))
    for idx, name in enumerate(meta["fields"]):
        structured[name] = columns[idx]
    return structured


def ExtractField(structured, name, default=None):
    if name in structured.dtype.names:
        return np.asarray(structured[name], dtype=np.float64).reshape(-1)
    if default is None:
        raise KeyError(name)
    return np.full(structured.shape[0], float(default), dtype=np.float64)


def _RequireCv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for image export but is not installed in the current environment.") from exc
    return cv2


def SaveColorizedLayer(filename, normalized_map, bgr_colormap=None):
    cv2 = _RequireCv2()
    if bgr_colormap is None:
        bgr_colormap = cv2.COLORMAP_TURBO
    EnsureDir(os.path.dirname(filename))
    valid_mask = np.isfinite(normalized_map)
    image = np.zeros(normalized_map.shape + (3,), dtype=np.uint8)
    if np.any(valid_mask):
        uint_map = np.zeros(normalized_map.shape, dtype=np.uint8)
        uint_map[valid_mask] = np.clip(normalized_map[valid_mask] * 255.0, 0, 255).astype(np.uint8)
        image = cv2.applyColorMap(uint_map, bgr_colormap)
        image[~valid_mask] = 0
    cv2.imwrite(filename, image)


def SaveLabelImage(filename, label_map, color_map):
    cv2 = _RequireCv2()
    EnsureDir(os.path.dirname(filename))
    image = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    for label_value, color in color_map.items():
        image[label_map == label_value] = np.asarray(color, dtype=np.uint8)
    cv2.imwrite(filename, image)


def SaveJson(filename, data):
    EnsureDir(os.path.dirname(filename))
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def BuildDefaultArgs(project_root):
    global_ini = ReadIniFile(os.path.join(project_root, "bin", "parameters", "Global.ini"))
    analyzer_ini = ReadIniFile(os.path.join(project_root, "bin", "parameters", "Mapper2DAnalyzer.ini"))
    mapper_ini = ReadIniFile(os.path.join(project_root, "bin", "parameters", "Mapper2D.ini"))

    args = Namespace(
        data_path=global_ini.get("m_LoadPath", ""),
        output_dir="",
        map_resolution=0.2,
        small_map_rows=200,
        big_map_rows=1400,
        roi_range=float(mapper_ini.get("ROI_RANGE", 80.0)),
        roi_vehicle_width_m=2.0,
        sampling_distance=float(mapper_ini.get("sampling_distance", 1.0)),
        k_valid_invalid_thresh=float(analyzer_ini.get("k_valid_invalid_thresh", 0.1)),
        slope_angle_thresh=float(analyzer_ini.get("slope_angle_thresh", 15.0)),
        bgk_radius=1.5,
        bgk_min_valid=5,
        bgk_bilateral_std=0.05,
        refine_passes=2,
        pose_sigma=0.08,
        sensor_sigma=0.03,
        std_floor=0.05,
        count_sigma_scale=8.0,
        local_span_sigma_scale=0.5,
        local_span_thresh=0.2,
        local_ground_window=2,
        local_height_thresh=0.25,
        local_slope_thresh=0.35,
        base_sensor_height=1.5,
        min_ground_weight=0.15,
        acceptance_k_sigma=3.0,
        acceptance_margin=0.08,
        negative_k_sigma=1.5,
        structure_bins="-1.0,-0.3,0.0,0.3,1.0,2.0,4.0",
        b_use_map3d=True,
        export_profile="full",
        b_reset_map2d=True,
        b_save_to_disk=True,
        b_load_from_disk=False,
        ground_backend="heuristic",
        patchwork_cli_path="",
        b_use_patchwork_server=True,
        b_visualize_intermediate=True,
        debug_frame_stride=5,
        debug_map_stride=5,
        debug_refine_stride=1,
        debug_max_frames=24,
    )
    if not args.output_dir:
        args.output_dir = os.path.join(args.data_path, "tmp_data", "python_terrain")
    return args


def BuildMapInfoPath(data_path):
    return os.path.join(data_path, "tmp_data", "map2D_analyzer", "map_info.txt")


def LoadMapInfo(filename):
    with open(filename, "r", encoding="utf-8") as fp:
        values = [int(item.strip()) for item in fp.readlines()[:4]]
    return values[0], values[1], values[2], values[3]


def SaveMapInfo(filename, map_min_x, map_min_y, rows, cols):
    with open(filename, "w", encoding="utf-8") as fp:
        fp.write(f"{map_min_x}\n{map_min_y}\n{rows}\n{cols}\n")


def NormalizeForImage(layer_map, invalid_value=INVALID_HEIGHT):
    valid_mask = np.isfinite(layer_map) & (layer_map != invalid_value)
    normalized = np.full(layer_map.shape, np.nan, dtype=np.float32)
    if np.any(valid_mask):
        valid_values = layer_map[valid_mask]
        min_value = float(np.min(valid_values))
        max_value = float(np.max(valid_values))
        if max_value > min_value:
            normalized[valid_mask] = ((valid_values - min_value) / (max_value - min_value)).astype(np.float32)
        else:
            normalized[valid_mask] = 0.0
    return normalized
