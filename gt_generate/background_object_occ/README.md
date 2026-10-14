# Background Object OCC

这个目录用于生成静态背景物体 OCC。

核心逻辑：

1. 读取 Waymo sequence 的每帧 `output/pcd/*.pcd`，点云输入为当前帧车体 / LiDAR 坐标系 `[N, 3/4]`。
2. 读取同 timestamp 的人工语义标签 `output/SemanticSegmentation/*.npy`；如果没有人工标签，则读取 `output/label/pseudo_label_2/*.label`。
3. 人工标签是原始 Waymo 标签，会先通过 `Scene_Tag_Completion/label_mapping/waymo_merge_example.yaml` 映射到当前 9 类训练标签；`pseudo_label_2` 本身已经是映射后的标签。
4. 读取阶段先过滤标签值为 `0` 的点；不做 `pc_range` 裁剪。
5. 每帧先过滤 `UNKNOWN, VEHICLE, CYCLIST, PEDESTRIAN, ROAD, LOW_GROUND`，保留静态背景物体类别。
6. 用 pose `timestamp x y z roll pitch yaw` 把静态点从车体坐标系变换到世界坐标系，并流式累积到 `0.05m` 中间 voxel map，避免保存全量原始拼接点。
7. 对 `0.05m` 中间静态地图按类别分别聚类，删除小簇零星虚点。
8. 对聚类后的静态地图按正式 OCC voxel size 聚合并做类别投票。一个 voxel 内如果存在人工标签投票，则只用人工标签多数类；否则使用伪标签多数类。
9. 将最终全局静态地图按每帧 pose 变换回当前帧车体 / LiDAR 坐标系，用 `pc_range` 裁剪后输出每帧局部 sparse OCC。

默认输出：

```bash
sequence/output/label/background_occ/
  <timestamp>.npz
  frame_occ_summary.json
```

每个 `<timestamp>.npz` 字段：

- `voxel_indices`: shape `[M, 3]`，当前帧车体 / LiDAR 坐标系下的局部 int32 voxel index。
- `labels`: shape `[M]`，9 类训练标签 id。
- `sources`: shape `[M]`，`1` 表示人工标签优先得到，`2` 表示伪标签得到。
- `local_voxel_centers_xyz`: shape `[M, 3]`，当前帧车体 / LiDAR 坐标系下 voxel center。
- `voxel_size_xyz`, `pc_range`, `grid_shape`: 定义每帧局部 OCC 网格。
- `timestamp`: 当前帧时间戳。
- `empty_label`: 固定为 `0`。

单序列运行示例：

```bash
python gt_generate/background_object_occ/generate_background_object_occ.py \
  data_path=$WAYMO_DATA_ROOT \
  sequence_name=16102220208346880_1420_000_1440_000 \
  voxel_size_xyz=0.2,0.2,0.2
```

调试时输出到临时目录：

```bash
python gt_generate/background_object_occ/generate_background_object_occ.py \
  data_path=$WAYMO_DATA_ROOT \
  sequence_name=16102220208346880_1420_000_1440_000 \
  output_dir=/tmp/background_object_occ_debug \
  max_frames=5
```

批处理全部序列：

```bash
python gt_generate/background_object_occ/generate_background_object_occ.py \
  data_path=$WAYMO_DATA_ROOT \
  sequence_name=all
```
