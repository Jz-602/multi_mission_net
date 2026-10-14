"""寻找70米内车辆失去检测标注的候选样本，只打印，不判断遮挡或补框。"""
import argparse
import math
import os
import pickle
from pathlib import Path


DEFAULT_SEQUENCE = "17642771458376274038_2080_000_2100_000"
MAX_DISTANCE_M = 50.0


def FindDisappearances(frames, max_distance_m=MAX_DISTANCE_M):
    # frames 为检测帧列表；id 是跨帧检测目标 ID，仅筛选 VEHICLE=1。
    # box [7]: 当前帧 LiDAR 系的 cx,cy,cz,length,width,height,yaw。
    # speed [3]: 原始检测数据中的世界系速度，单位 m/s，仅展示，不做外推。
    frames = sorted(frames, key=lambda frame: frame["frame_timestamp_micros"])
    tracks = {}
    for frame_index, frame in enumerate(frames):
        for obj in frame["objects"]:
            if int(obj["type"]) == 1:
                tracks.setdefault(str(obj["id"]), []).append((frame_index, obj))

    candidates = []
    for object_id, observations in tracks.items():
        for observation_index, (frame_index, obj) in enumerate(observations):
            next_frame = observations[observation_index + 1][0] if observation_index + 1 < len(observations) else None
            end_frame = next_frame if next_frame is not None else len(frames)
            missing_count = end_frame - frame_index - 1
            distance_m = math.hypot(float(obj["box"][0]), float(obj["box"][1]))
            if missing_count == 0 or distance_m > max_distance_m:
                continue
            candidates.append({
                "id": object_id,
                "kind": "消失后再出现" if next_frame is not None else "消失后未再出现",
                "first_frame": observations[0][0],
                "last_seen_frame": frame_index,
                "last_seen_timestamp_ms": int(frames[frame_index]["frame_timestamp_micros"]) // 1000,
                "missing_start_frame": frame_index + 1,
                "missing_end_frame": end_frame - 1,
                "missing_frame_count": missing_count,
                "next_seen_frame": next_frame,
                "distance_m": distance_m,
                "last_box": [float(value) for value in obj["box"]],
                "last_speed_world_mps": [float(value) for value in obj["speed"]],
            })
    return candidates


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("WAYMO_DATA_ROOT", ""))
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE, help="序列名，all 表示遍历数据根目录下已有序列")
    parser.add_argument("--max-distance-m", type=float, default=MAX_DISTANCE_M)
    args = parser.parse_args()
    root = Path(args.data_root)
    sequence_dirs = sorted(path for path in root.iterdir() if (path / "output" / "pcd").is_dir()) if args.sequence == "all" else [root / args.sequence]
    total_candidates = 0
    matched_sequences = 0
    print(f"条件：车辆最后可见框中心水平距离 <= {args.max_distance_m:g} 米，后续至少一帧缺少同一 ID；帧号从 0 开始。")
    print("分别列出再次出现和未再出现的候选；序列结束不算消失，不设置缺失帧数上限。")
    for sequence_dir in sequence_dirs:
        with (sequence_dir / "output" / "detection_info.pkl").open("rb") as file:
            frames = pickle.load(file)
        candidates = FindDisappearances(frames, args.max_distance_m)
        returned = sum(item["next_seen_frame"] is not None for item in candidates)
        print(f"\n序列={sequence_dir.name}，帧数={len(frames)}，消失后再出现={returned}，消失后未再出现={len(candidates) - returned}")
        for item in candidates:
            box = item["last_box"]
            speed = item["last_speed_world_mps"]
            print(
                f"[{item['kind']}] ID={item['id']}，首次第 {item['first_frame']} 帧，"
                f"最后可见第 {item['last_seen_frame']} 帧（时间戳 {item['last_seen_timestamp_ms']}），"
                f"缺失第 {item['missing_start_frame']}–{item['missing_end_frame']} 帧，共 {item['missing_frame_count']} 帧；"
                f"再次出现帧={item['next_seen_frame']}，最后距离={item['distance_m']:.1f} 米，"
                f"位置=({box[0]:.2f},{box[1]:.2f},{box[2]:.2f}) 米，"
                f"世界系速度=({speed[0]:.2f},{speed[1]:.2f},{speed[2]:.2f}) m/s，"
                f"水平速率={math.hypot(speed[0], speed[1]):.2f} m/s"
            )
        total_candidates += len(candidates)
        matched_sequences += bool(candidates)
    print(f"\n合计：检查 {len(sequence_dirs)} 段，有候选 {matched_sequences} 段，候选事件 {total_candidates} 个。")


if __name__ == "__main__":
    Main()
