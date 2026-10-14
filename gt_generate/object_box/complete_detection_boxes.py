"""三类检测框首次出现前、消失后的双向补框实验；只返回内存结果，不改写 detection_info.pkl。

运行：python gt_generate/object_box/complete_detection_boxes.py --data-root "$WAYMO_DATA_ROOT"
默认用锚点相邻5个连续观测帧的世界系速度估计水平加速度，仅在地图支持区域内、70米范围内补框。
补充框中心到原始锚点的世界系水平距离最多2米；首次出现前使用出现后的观测回推。
--motion-model velocity 切回匀速；范围按当前自车中心计算，不设时间上限。
地图支持是实验中的地面覆盖约束，不代表已确认遮挡或车辆可通行。
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.io_utils import LoadMapInfo, ReadMat


DEFAULT_SEQUENCE = "4457475194088194008_3100_000_3120_000"
TYPE_NAMES = {1: "VEHICLE", 2: "PEDESTRIAN", 4: "CYCLIST"}
# 全局 terrain_ground_type_map 的编码：1=ROAD，2=LOW_GROUND。
# 第一版实验方案：车辆采用 ROAD；行人、骑行者允许 LOW_GROUND。
SUPPORTED_GROUND_TYPES = {1: (1,), 2: (1, 2), 4: (1, 2)}
STOP_NAMES = {
    "range": "超过当前自车距离上限",
    "anchor_distance": "超过原始锚点距离上限",
    "outside_map": "外推路径离开地图支持区域",
    "observed_again": "接回原始检测框",
    "sequence_end": "序列结束",
    "sequence_start": "序列开始",
}


def BuildParser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", default=os.environ.get("WAYMO_DATA_ROOT", ""))
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--max-distance-m", type=float, default=70.0)
    parser.add_argument("--max-anchor-distance-m", type=float, default=2.0, help="距原始锚点的世界系水平距离上限，米")
    parser.add_argument("--motion-model", choices=("velocity", "acceleration"), default="acceleration")
    parser.add_argument("--history-frames", type=int, default=5, help="加速度估计所用的锚点相邻连续观测帧数")
    parser.add_argument("--max-acceleration-mps2", type=float, default=4.0, help="水平加速度模长上限，m/s²")
    return parser


class TerrainSupport:
    def __init__(self, terrain_dir):
        terrain_dir = Path(terrain_dir)
        self.height = ReadMat(str(terrain_dir / "terrain_z_inpainted.bin"))
        self.ground_type = ReadMat(str(terrain_dir / "terrain_ground_type_map.bin"))
        min_x_cm, min_y_cm, rows, cols = LoadMapInfo(str(terrain_dir / "map_info.txt"))
        with (terrain_dir / "terrain_only_summary.json").open() as file:
            summary = json.load(file)
        self.resolution_m = float(summary["map_resolution_m"])
        self.min_xy = np.asarray([min_x_cm, min_y_cm], dtype=np.float64) / 100.0
        if self.height.shape != (rows, cols) or self.ground_type.shape != (rows, cols):
            raise ValueError("全局地形高度、类别图与 map_info 尺寸不一致")

    def Contains(self, world_points, type_id):
        """world_points [N,3] 为世界系位置；按框中心的 XY 所在栅格判断支持。"""
        world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        cells = np.floor((world_points[:, :2] - self.min_xy) / self.resolution_m).astype(np.int64)
        rows = self.height.shape[0] - 1 - cells[:, 1]
        cols = cells[:, 0]
        b_inside = (rows >= 0) & (cols >= 0) & (rows < self.height.shape[0]) & (cols < self.height.shape[1])
        selected = np.flatnonzero(b_inside)
        z = self.height[rows[selected], cols[selected]]
        ground_type = self.ground_type[rows[selected], cols[selected]]
        b_inside[selected] = np.isfinite(z) & (z != INVALID_HEIGHT) & np.isin(ground_type, SUPPORTED_GROUND_TYPES[type_id])
        return b_inside

    def SupportsPath(self, start_world, end_world, type_id):
        """检查世界系端点 [3] 及线段中点采样是否均受地图支持。

        按半个栅格间距采样，避免两端在地图内却跨过图外区域。
        """
        if not np.all(self.Contains(np.stack([start_world, end_world]), type_id)):
            return False
        distance_m = float(np.linalg.norm((end_world - start_world)[:2]))
        if distance_m == 0.0:
            return True
        sample_num = max(1, int(np.ceil(distance_m / (0.5 * self.resolution_m))))
        fractions = (np.arange(sample_num) + 0.5) / sample_num
        samples = start_world + fractions[:, None] * (end_world - start_world)
        return bool(np.all(self.Contains(samples, type_id)))

    def WorldPoints(self, stride=2):
        """返回全局地图世界系 [N,3] 格心及 [N] 地面类别，供可视化复用。"""
        rows, cols = np.indices(self.height.shape)
        b_valid = np.isfinite(self.height) & (self.height != INVALID_HEIGHT) & (self.ground_type > 0)
        b_valid &= (rows % stride == 0) & (cols % stride == 0)
        rows, cols = np.nonzero(b_valid)
        x = self.min_xy[0] + (cols + 0.5) * self.resolution_m
        y = self.min_xy[1] + (self.height.shape[0] - rows - 0.5) * self.resolution_m
        return np.column_stack([x, y, self.height[rows, cols]]), self.ground_type[rows, cols]


def EstimateAcceleration(frames, observations, observation_idx, history_frames, max_acceleration_mps2, observation_step=-1):
    """用锚点连续观测的世界系速度 [K,3] 拟合水平加速度 [3]。

    observation_step=-1 使用消失前历史；+1 使用首次出现后的观测作离线回推。
    少于3帧时返回零加速度；Z不加速。模长限幅是实验参数，不是道路先验。
    """
    history = []
    expected_idx = observations[observation_idx][0]
    type_id = observations[observation_idx][1]["type"]
    for offset in range(history_frames):
        index = observation_idx + offset * observation_step
        if not 0 <= index < len(observations):
            break
        frame_idx, obj = observations[index]
        if frame_idx != expected_idx or obj["type"] != type_id:
            break
        history.append((int(frames[frame_idx]["frame_timestamp_micros"]), np.asarray(obj["speed"], dtype=np.float64)))
        expected_idx += observation_step
    acceleration = np.zeros(3, dtype=np.float64)
    if len(history) >= 3:
        times = np.asarray([item[0] - history[0][0] for item in history], dtype=np.float64) * 1e-6
        speeds = np.stack([item[1][:2] for item in history])
        if not np.all(np.isfinite(speeds)):
            raise ValueError("历史速度中存在非有限数值")
        centered_time = times - times.mean()
        acceleration[:2] = np.sum(centered_time[:, None] * (speeds - speeds.mean(axis=0)), axis=0) / np.sum(centered_time ** 2)
        magnitude = np.linalg.norm(acceleration[:2])
        if magnitude > max_acceleration_mps2:
            acceleration *= max_acceleration_mps2 / magnitude
    return acceleration, len(history)


def PredictMotion(center_world, speed_world, acceleration_world, heading_world, elapsed_s):
    """世界系匀加速位置与朝向。框保持原始朝向和运动方向的夹角。

    仅当初始、预测水平速率都超过0.5m/s时更新朝向，避免低速速度抖动引起转框。
    """
    predicted_world = center_world + speed_world * elapsed_s + 0.5 * acceleration_world * elapsed_s ** 2
    predicted_speed = speed_world + acceleration_world * elapsed_s
    predicted_heading = heading_world.copy()
    if np.linalg.norm(speed_world[:2]) > 0.5 and np.linalg.norm(predicted_speed[:2]) > 0.5:
        cross = speed_world[0] * predicted_speed[1] - speed_world[1] * predicted_speed[0]
        delta = np.arctan2(cross, np.dot(speed_world[:2], predicted_speed[:2]))
        c, s = np.cos(delta), np.sin(delta)
        predicted_heading[:2] = [[c, -s], [s, c]] @ heading_world[:2]
    return predicted_world, predicted_heading


def CompleteDetectionBoxes(frames, terrain, max_distance_m=70.0,
                           motion_model="acceleration", history_frames=5, max_acceleration_mps2=4.0,
                           max_anchor_distance_m=2.0):
    """原始 frames 只读；首次出现前回推，原始观测中断后外推。

    输入 box [7] 为当前帧 LiDAR 系 cx,cy,cz,l,w,h,yaw；pose [4,4] 是
    world_from_lidar，speed [3] 为世界系 m/s。锚点为原始框世界系中心 [3]，
    每个事件固定不变；距离限制为世界系 XY 直线距离，不累计路程或重置锚点。
    输出 box 转回目标帧 LiDAR 系，沿用仅 yaw 的原始框格式。
    """
    if not np.all(np.isfinite([max_distance_m, max_anchor_distance_m])) or min(max_distance_m, max_anchor_distance_m) <= 0.0:
        raise ValueError("距离阈值必须为有限正数")
    if motion_model not in ("velocity", "acceleration") or history_frames < 3 or max_acceleration_mps2 <= 0:
        raise ValueError("运动模式、历史帧数或加速度上限无效")
    frames = sorted(frames, key=lambda frame: frame["frame_timestamp_micros"])
    supplements = [[] for _ in frames]
    tracks = {}
    for frame_idx, frame in enumerate(frames):
        for obj in frame["objects"]:
            # 所有原始 ID 都参与接回判定；只有三任务检测类别生成补框。
            tracks.setdefault(str(obj["id"]), []).append((frame_idx, obj))

    events = []
    for object_id, observations in tracks.items():
        # 内部缺口由前向处理；后向只覆盖首次出现前，避免同一 ID 双向重复补框。
        candidates = []
        if observations[0][0] > 0:
            candidates.append((0, -1, -1))
        for observation_idx, (anchor_idx, _) in enumerate(observations):
            next_idx = observations[observation_idx + 1][0] if observation_idx + 1 < len(observations) else len(frames)
            if next_idx > anchor_idx + 1:
                candidates.append((observation_idx, 1, next_idx))
        for observation_idx, time_step, boundary_idx in candidates:
            anchor_idx, obj = observations[observation_idx]
            type_id = int(obj["type"])
            if type_id not in TYPE_NAMES:
                continue
            box = np.asarray(obj["box"], dtype=np.float64)
            anchor_pose = np.asarray(frames[anchor_idx]["pose"], dtype=np.float64)
            speed_world = np.asarray(obj["speed"], dtype=np.float64)
            if box.shape != (7,) or speed_world.shape != (3,) or not np.all(np.isfinite(np.r_[box, speed_world])):
                raise ValueError(f"无效检测框或速度：frame={anchor_idx}, id={object_id}")
            if np.linalg.norm(box[:2]) > max_distance_m:
                continue
            center_world = anchor_pose[:3, :3] @ box[:3] + anchor_pose[:3, 3]
            heading_world = anchor_pose[:3, :3] @ np.asarray([np.cos(box[6]), np.sin(box[6]), 0.0])
            acceleration_world = np.zeros(3, dtype=np.float64)
            history_count = 1
            if motion_model == "acceleration":
                acceleration_world, history_count = EstimateAcceleration(
                    frames, observations, observation_idx, history_frames, max_acceleration_mps2,
                    observation_step=-time_step,
                )
            anchor_time = int(frames[anchor_idx]["frame_timestamp_micros"])
            previous_world = center_world.copy()
            b_backward = time_step < 0
            stop_reason = "sequence_start" if b_backward else ("observed_again" if boundary_idx < len(frames) else "sequence_end")
            event = {
                "event_index": len(events), "id": object_id, "type": type_id,
                "direction": "backward" if b_backward else "forward",
                "anchor_frame": anchor_idx, "anchor_world": center_world.copy(),
                "last_seen_frame": None if b_backward else anchor_idx,
                "next_seen_frame": anchor_idx if b_backward else (boundary_idx if boundary_idx < len(frames) else None),
                "supplemented_frame_count": 0, "stop_frame": max(0, min(boundary_idx, len(frames) - 1)),
                "stop_reason": stop_reason,
                "acceleration_world": acceleration_world, "history_frames": history_count,
            }
            for frame_idx in range(anchor_idx + time_step, boundary_idx, time_step):
                # 负时间直接代入运动方程；回推时不反转目标朝向。
                elapsed_s = (int(frames[frame_idx]["frame_timestamp_micros"]) - anchor_time) * 1e-6
                predicted_world, predicted_heading = PredictMotion(center_world, speed_world, acceleration_world, heading_world, elapsed_s)
                anchor_distance_m = float(np.linalg.norm((predicted_world - center_world)[:2]))
                if anchor_distance_m > max_anchor_distance_m:
                    event.update(stop_frame=frame_idx, stop_reason="anchor_distance")
                    break
                pose = np.asarray(frames[frame_idx]["pose"], dtype=np.float64)
                local_center = pose[:3, :3].T @ (predicted_world - pose[:3, 3])
                distance_m = float(np.linalg.norm(local_center[:2]))
                if distance_m > max_distance_m:
                    event.update(stop_frame=frame_idx, stop_reason="range")
                    break
                if not terrain.SupportsPath(previous_world, predicted_world, type_id):
                    event.update(stop_frame=frame_idx, stop_reason="outside_map")
                    break
                local_heading = pose[:3, :3].T @ predicted_heading
                local_box = np.r_[local_center, box[3:6], np.arctan2(local_heading[1], local_heading[0])]
                supplements[frame_idx].append({
                    "id": object_id, "type": type_id, "box": local_box,
                    "event_index": event["event_index"], "elapsed_s": elapsed_s,
                    "direction": event["direction"], "anchor_distance_m": anchor_distance_m,
                    "distance_m": distance_m,
                })
                event["supplemented_frame_count"] += 1
                previous_world = predicted_world
            events.append(event)
    return frames, supplements, events


def RunExperiment(args):
    if not args.data_root:
        raise ValueError("请设置 WAYMO_DATA_ROOT 或传入 --data-root")
    sequence_root = Path(args.data_root).expanduser() / args.sequence
    with (sequence_root / "output/detection_info.pkl").open("rb") as file:
        frames = pickle.load(file)
    terrain = TerrainSupport(sequence_root / "output/label/ground_truth")
    frames, supplements, events = CompleteDetectionBoxes(frames, terrain, args.max_distance_m,
                                                        args.motion_model, args.history_frames, args.max_acceleration_mps2, args.max_anchor_distance_m)
    return {
        "sequence_root": sequence_root, "terrain": terrain,
        "frames": frames, "supplements": supplements, "events": events, "motion_model": args.motion_model, "max_anchor_distance_m": args.max_anchor_distance_m,
    }


def PrintSummary(result):
    counts = Counter(TYPE_NAMES[obj["type"]] for items in result["supplements"] for obj in items)
    direction_counts = Counter(obj["direction"] for items in result["supplements"] for obj in items)
    print(f"序列：{result['sequence_root'].name}；{len(result['frames'])} 帧；原始检测文件只读。")
    print(f"运动模式：{result['motion_model']}；距原始锚点最多 {result['max_anchor_distance_m']:g} 米")
    print(f"候选事件 {len(result['events'])} 个；补充框总计 {sum(counts.values())} 个（按目标帧计）：{dict(counts)}")
    print(f"首次出现前回推 {direction_counts['backward']} 个；消失后外推 {direction_counts['forward']} 个。")
    print("以下为双向补框实验，未自动确认遮挡或 ID 切换；仅在地图支持区域内补框，不设时间上限。")
    for event in result["events"]:
        reason = STOP_NAMES[event["stop_reason"]]
        print(f"{TYPE_NAMES[event['type']]} id={event['id']} {event['direction']} anchor={event['anchor_frame']} "
              f"add={event['supplemented_frame_count']} stop={event['stop_frame']} "
              f"reason={reason} "
              f"a_xy={np.round(event['acceleration_world'][:2], 2)} history={event['history_frames']}")


if __name__ == "__main__":
    PrintSummary(RunExperiment(BuildParser().parse_args()))
