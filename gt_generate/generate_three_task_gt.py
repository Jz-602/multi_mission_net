"""按序列生成地形与静态 OCC，核对复用的检测标注；每个步骤独立进程运行。"""
import argparse
import fcntl
import json
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path
from zipfile import BadZipFile

sys.dont_write_bytecode = True

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from gt_generate.common.io_utils import LoadMapInfo, LoadPoseDb
from gt_generate.ground_truth.data_paths import ListFhSequenceDirs, ResolveTerrainOnlyDataPaths

STAGES = (
    ("terrain_global", "gt_generate/ground_truth/build_waymo_terrain_only.py", []),
    ("terrain_frames", "gt_generate/ground_truth/export_frame_terrain_gt.py", ["b_clip_to_last_semantic_frame=False"]),
    ("background_occ", "gt_generate/background_object_occ/generate_background_object_occ.py", []),
)
TERRAIN_FIELDS = ("terrain_z_inpainted", "terrain_inpainted_mask", "terrain_inpaint_distance", "terrain_train_weight")


def LoadTargetTimestamps(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    sequence_timestamps = {}
    for split in manifest["splits"].values():
        for target in split["targets"]:
            sequence_timestamps.setdefault(str(target["sequence_name"]), set()).add(int(target["timestamp_ms"]))
    if not sequence_timestamps:
        raise ValueError("Target manifest is empty")
    return sequence_timestamps


def ValidateInputs(sequence_dir):
    args = argparse.Namespace(data_path=str(sequence_dir), sequence_name="")
    paths = ResolveTerrainOnlyDataPaths(args, b_require_semantic=True, b_require_pose=True)
    timestamps = {int(p.stem) for p in Path(paths.pcd_dir).glob("*.pcd")}
    if not timestamps:
        raise ValueError(f"No point cloud frames: {sequence_dir}")
    pose_timestamps, _ = LoadPoseDb(paths.pose_filename)
    if not timestamps.issubset({int(t) for t in pose_timestamps}):
        raise ValueError(f"Missing frame poses: {sequence_dir}")
    output = sequence_dir / "output"
    for timestamp in timestamps:
        label = output / "label/pseudo_label_2" / f"{timestamp}.label"
        if not label.is_file() or label.stat().st_size == 0 or label.stat().st_size % 4:
            raise ValueError(f"Missing or invalid uint32 pseudo label: {label}")
    # 检测记录按毫秒时间戳匹配当前帧；框沿用原始车体/LiDAR系标注，不做补框或变换。
    with (output / "detection_info.pkl").open("rb") as stream:
        detection = pickle.load(stream)
    detection_timestamps = {int(item["frame_timestamp_micros"]) // 1000 for item in detection}
    if not timestamps.issubset(detection_timestamps):
        raise ValueError(f"Missing detection frames: {sequence_dir}")
    return timestamps


def ValidateBin(path, expected_shape=None):
    with path.open("rb") as stream:
        rows, cols, cv_type = struct.unpack("<iii", stream.read(12))
    itemsize = {0: 1, 2: 2, 4: 4, 5: 4, 6: 8}[cv_type & 7]
    channels = 1 + (cv_type >> 3)
    if rows <= 0 or cols <= 0 or path.stat().st_size != 12 + rows * cols * channels * itemsize:
        raise ValueError(f"Invalid terrain matrix: {path}")
    if expected_shape is not None and (rows, cols) != expected_shape:
        raise ValueError(f"Unexpected terrain shape: {path}")


def ValidateOutputs(sequence_dir, timestamps, stage, b_target_subset=False):
    output = sequence_dir / "output/label"
    terrain = output / "ground_truth"
    if stage == "terrain_global":
        LoadMapInfo(str(terrain / "map_info.txt"))
        for field in ("terrain_z_final_seed", "terrain_z_inpainted", "terrain_inpaint_distance"):
            ValidateBin(terrain / f"{field}.bin")
        json.loads((terrain / "terrain_only_summary.json").read_text())
    elif stage == "terrain_frames":
        summary = json.loads((terrain / "terrain_only_frame_gt_manifest.json").read_text())
        exported_timestamps = {f["timestamp"] for f in summary["frames"]}
        # 完整存量可覆盖目标子集，直接复用，避免缩小已有摘要或删除额外帧。
        b_coverage_matches = timestamps.issubset(exported_timestamps) if b_target_subset else exported_timestamps == timestamps
        if summary["clip_to_last_semantic_frame"] or not b_coverage_matches:
            raise ValueError(f"Terrain frame coverage mismatch: {sequence_dir}")
        for timestamp in timestamps:
            frame = terrain / "frames" / f"{timestamp:013d}"
            metadata = json.loads((frame / "frame_meta.json").read_text())
            if metadata["timestamp"] != timestamp:
                raise ValueError(f"Terrain timestamp mismatch: {frame}")
            for field in TERRAIN_FIELDS:
                ValidateBin(frame / "base/terrain_completion" / f"{field}.bin", (800, 800))
    else:
        occ = output / "background_occ"
        summary = json.loads((occ / "frame_occ_summary.json").read_text())
        exported_timestamps = {f["timestamp"] for f in summary["frame_occ"]}
        b_coverage_matches = timestamps.issubset(exported_timestamps) if b_target_subset else exported_timestamps == timestamps
        if not b_coverage_matches:
            raise ValueError(f"OCC frame coverage mismatch: {sequence_dir}")
        # OCC 坐标为当前帧车体系局部网格 (N,3)，语义标签 (N,)。
        for timestamp in timestamps:
            with np.load(occ / f"{timestamp}.npz") as data:
                indices, labels, shape = data["voxel_indices"], data["labels"], data["grid_shape"]
                if indices.shape != (len(labels), 3) or np.any(indices < 0) or np.any(indices >= shape):
                    raise ValueError(f"Invalid sparse OCC: {sequence_dir.name}/{timestamp}")
                if not np.allclose(data["voxel_size_xyz"], 0.2) or not np.allclose(data["pc_range"], [-80, -80, -7, 80, 80, 5.8]):
                    raise ValueError(f"Unexpected OCC grid: {sequence_dir.name}/{timestamp}")


def RunStage(sequence_dir, timestamps, stage, script, extra_args, b_overwrite=False, target_timestamps=None):
    b_target_subset = target_timestamps is not None and stage != "terrain_global"
    expected_timestamps = target_timestamps if b_target_subset else timestamps
    if not b_overwrite:
        try:
            ValidateOutputs(sequence_dir, expected_timestamps, stage, b_target_subset)
        except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError, struct.error, BadZipFile):
            pass
        else:
            print(f"SKIP {sequence_dir.name} {stage}", flush=True)
            return

    # 生成前使旧完成摘要失效，避免中断后把新旧混合文件误认为完整结果。
    terrain = sequence_dir / "output/label/ground_truth"
    summaries = {
        "terrain_global": [terrain / "terrain_only_summary.json", terrain / "terrain_only_frame_gt_manifest.json", terrain / "terrain_only_frame_gt_summary.json"],
        "terrain_frames": [terrain / "terrain_only_frame_gt_manifest.json", terrain / "terrain_only_frame_gt_summary.json"],
        "background_occ": [sequence_dir / "output/label/background_occ/frame_occ_summary.json"],
    }
    for summary in summaries[stage]:
        summary.unlink(missing_ok=True)
    command = [sys.executable, "-B", str(PROJECT_ROOT / script), f"data_path={sequence_dir}", "sequence_name="] + extra_args
    if b_target_subset:
        command.append("target_timestamps=" + ",".join(str(timestamp) for timestamp in sorted(target_timestamps)))
    print(f"START {sequence_dir.name} {stage}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    ValidateOutputs(sequence_dir, expected_timestamps, stage, b_target_subset)
    print(f"DONE {sequence_dir.name} {stage} frames={len(expected_timestamps)}", flush=True)


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("WAYMO_DATA_ROOT", ""))
    parser.add_argument("--sequence", help="仅运行一个序列；默认处理根目录下所有序列")
    parser.add_argument("--overwrite", dest="b_overwrite", action="store_true", help="重新生成完整结果；输入或生成规则改变时使用")
    parser.add_argument("--check-only", dest="b_check_only", action="store_true", help="只核对输入，不生成真值")
    parser.add_argument("--target-manifest", help="仅导出清单 splits.targets 的监督帧；完整序列仍用于建图")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("Set WAYMO_DATA_ROOT or pass --data-root")
    sequences = [Path(p) for p in ListFhSequenceDirs(args.data_root)]
    if args.sequence:
        sequences = [p for p in sequences if p.name == args.sequence]
    target_timestamps = LoadTargetTimestamps(args.target_manifest) if args.target_manifest else None
    if target_timestamps is not None:
        requested_sequences = {args.sequence} if args.sequence else set(target_timestamps)
        if not requested_sequences.issubset(target_timestamps):
            parser.error("Requested sequence has no targets in manifest")
        sequences = [p for p in sequences if p.name in requested_sequences]
        if {p.name for p in sequences} != requested_sequences:
            parser.error("Target manifest contains missing sequences")
    if not sequences:
        parser.error("No matching sequences")
    inputs = {p: ValidateInputs(p) for p in sequences}
    if target_timestamps is not None:
        for sequence_dir, timestamps in inputs.items():
            if not target_timestamps[sequence_dir.name].issubset(timestamps):
                raise ValueError(f"Target timestamps missing point clouds: {sequence_dir.name}")
    print(f"INPUTS_OK sequences={len(inputs)} frames={sum(map(len, inputs.values()))}", flush=True)
    if target_timestamps is not None:
        print(f"TARGETS_OK frames={sum(len(target_timestamps[p.name]) for p in inputs)}", flush=True)
    if args.b_check_only:
        return
    # 对现有数据目录加进程锁，不创建锁文件；Linux 本机和服务器均可使用。
    lock_fd = os.open(args.data_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, (sequence_dir, timestamps) in enumerate(inputs.items(), start=1):
            print(f"SEQUENCE {index}/{len(inputs)} {sequence_dir.name}", flush=True)
            for stage, script, extra_args in STAGES:
                RunStage(sequence_dir, timestamps, stage, script, extra_args, args.b_overwrite,
                         None if target_timestamps is None else target_timestamps[sequence_dir.name])
        exported_frame_count = sum(map(len, inputs.values())) if target_timestamps is None else sum(len(target_timestamps[p.name]) for p in inputs)
        print(f"ALL_COMPLETE sequences={len(inputs)} frames={exported_frame_count}", flush=True)
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    Main()


# 运行命令：先进入 multi_mission 仓库根目录。
# conda activate multi_mission
# 主数据路径只在这里设置；本机使用本机目录，服务器使用下面的目录。
# export WAYMO_DATA_ROOT=/media/gk/data/lyf/aa_data/FH_waymo
# python -B gt_generate/generate_three_task_gt.py --check-only
# python -B gt_generate/generate_three_task_gt.py
# 中断后重复上一条命令，核对完整结果后跳过；输入或生成规则改变时加 --overwrite。
# 也可只为本次运行指定主路径：
# python -B gt_generate/generate_three_task_gt.py --data-root /media/gk/data/lyf/aa_data/FH_waymo
