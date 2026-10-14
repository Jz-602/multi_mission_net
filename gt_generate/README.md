# gt_generate

这个目录用于生成 Waymo 场景相关真值。目前主要包含三部分：

1. 地面 / 地形真值生成。
2. 静态背景物体 OCC 生成。
3. 将地面真值、背景 OCC、检测框组合起来做 Open3D 可视化复查。

后续目标框相关处理会继续放在这个项目里，但当前还没有纳入这里的主流程。

## 目录结构

- `common/`: 通用常量、IO、位姿/几何变换、Open3D 辅助函数和训练工具。
- `ground_truth/`: 地面真值生成，包括全局地形图和逐帧训练 GT。
- `background_object_occ/`: 静态背景物体 OCC 生成。
- `network_prediction/`: 当前帧 BEV 地形预测模型的训练和测试入口。
- `visualize_static_scene_gt.py`: 静态场景真值可视化脚本。

## 地面真值来源

`ground_truth/build_waymo_terrain_only.py` 会从逐帧地面点构建全局地形地图。有人工 `output/SemanticSegmentation/*.npy` 标注的帧会全部使用；这些帧里的原始 Waymo 标签会先通过 `Scene_Tag_Completion/label_mapping/waymo_merge_example.yaml` 映射到当前训练标签空间，再筛选地面类别。没有人工标注的帧可以使用 `output/label/pseudo_label_2/*.label`，该伪标签本身已经是映射后的训练标签空间，默认按 `0.5m` 位移间隔采样加入。

当前流程只把 `ROAD` 和 `LOW_GROUND` 当作地面类别。生成的训练目标仍然是地形高度图 `terrain_z_inpainted`，以及对应的 mask、weight 和质量辅助图，不是语义占据图。

默认输出目录为当前 sequence 的 `output/label/ground_truth/`；逐帧训练 GT 会继续放在该目录下的 `frames/` 子目录。

逐帧训练 GT 只导出训练代码会读取的字段，避免保存调试用 raw/reliable 辅助图导致体积过大：

```text
frames/<timestamp>/base/terrain_completion/
  terrain_z_inpainted.bin
  terrain_inpainted_mask.bin
  terrain_inpaint_distance.bin
  terrain_train_weight.bin
frames/<timestamp>/frame_meta.json
```

主要输出：

```text
sequence/output/label/ground_truth/
  terrain_z_inpainted.bin
  terrain_ground_type_map.bin
  terrain_inpaint_distance.bin
  terrain_only_summary.json
  terrain_only_frame_gt_manifest.json
  frames/<timestamp>/base/terrain_completion/terrain_z_inpainted.bin
```

## 背景 OCC

`background_object_occ/generate_background_object_occ.py` 用于生成逐帧静态背景物体 OCC。它读取每帧点云和语义标签，先过滤标签值为 `0` 的点，然后过滤掉：

```text
UNKNOWN, VEHICLE, CYCLIST, PEDESTRIAN, ROAD, LOW_GROUND
```

剩下的点作为静态背景物体点，变换到世界坐标系后流式累积到 `0.05m` 中间 voxel map，再按类别分别聚类去掉小簇零星虚点。人工语义标签优先级高于伪标签；没有人工标签的帧会读取 `output/label/pseudo_label_2/*.label`。最后把全局静态 voxel 地图按每帧 pose 转回当前帧车体 / LiDAR 坐标系，用 `pc_range` 裁剪后保存每帧局部 OCC。

默认 voxel size:

```text
[0.2, 0.2, 0.2] m
```

默认输出目录：

```text
sequence/output/label/background_occ/
  <timestamp>.npz
  frame_occ_summary.json
```

没有出现在每帧 sparse OCC 文件里的 voxel 视为 `EMPTY=0`。

## 可视化

`visualize_static_scene_gt.py` 会读取正式数据目录中的：

```text
output/pcd/
output/detection_info.pkl
output/label/ground_truth/frames/
output/label/background_occ/<timestamp>.npz
```

可视化内容包括当前帧点云、地面真值、背景 OCC 和检测 GT 框。启动参数 `1` 表示显示当前帧点云，`2` 表示只显示真值。

## 常用命令

**先确认磁盘预算。** 默认入口给所选序列的全部点云帧导出标签，默认800×800地形四张矩阵约8.32MB/帧；若有17,000帧，仅逐帧地形就约141.44GB，OCC、全局图和训练权重另计。先用 `--sequence 序列名` 试跑并测量，再决定批处理范围。空间不足时可传 `--target-manifest`，只导出监督目标；完整序列仍用于全局地形和静态 OCC 建图，历史帧输入不受筛选影响。服务器执行顺序见 [服务器交接第4节](../3mission_model/SERVER_CODEX_HANDOFF.md#4-环境标签准备与磁盘预算)。`--check-only` 仅检查输入，不验收派生标签输出。

空间预算通过后，批量生成当前 `WAYMO_DATA_ROOT` 下所有序列的三任务真值：

```bash
conda activate multi_mission
export WAYMO_DATA_ROOT=/media/gk/data/lyf/aa_data/FH_waymo
python -B gt_generate/generate_three_task_gt.py --check-only
python -B gt_generate/generate_three_task_gt.py
```

入口先核对点云、位姿、伪标签和检测标注的帧覆盖，再逐序列运行全局地形、逐帧地形、静态 OCC。检测直接复用 `output/detection_info.pkl`，补框不进入训练；逐帧地形导出覆盖全部有点云的帧。已完整的输出核对通过后跳过，缺失或不完整的步骤重新生成。

仅导出固定目标时，清单沿用 `build_label_manifest.py` 的 `splits.*.targets` 结构，每个目标含 `sequence_name` 和毫秒 `timestamp_ms`。默认只处理清单涉及的序列；可另加 `--sequence` 限定一条序列。输入覆盖仍检查完整序列，输出覆盖检查目标集合；已有全帧输出可以直接复用。不会删除已有标签，逐帧元信息保持原字段与完整序列帧索引。

```bash
python -B gt_generate/generate_three_task_gt.py --target-manifest 3mission_model/data_manifests/waymo_three_task_20260930.json --check-only
python -B gt_generate/generate_three_task_gt.py --target-manifest 3mission_model/data_manifests/waymo_three_task_20260930.json
```

每个步骤独立进程串行运行，进度和错误直接显示在终端。批处理不再创建运行目录、日志、耗时文件、完成标记或锁文件；底层生成器原有的真值、全局地形和元数据仍会保存。失败或中断后重用同一命令续跑，完成情况直接从真值及其原有摘要核对。输入或生成规则改变时加 `--overwrite` 重新生成。`--sequence 序列名` 可单独运行一个序列。主路径也可用 `--data-root 路径` 临时覆盖，服务器需使用已安装项目依赖的 `multi_mission` 环境。

生成全局地面真值：

```bash
python gt_generate/ground_truth/build_waymo_terrain_only.py \
  data_path=$WAYMO_DATA_ROOT \
  sequence_name=5144634012371033641_920_000_940_000
```

导出逐帧地面训练 GT：

```bash
python gt_generate/ground_truth/export_frame_terrain_gt.py \
  data_path=$WAYMO_DATA_ROOT \
  sequence_name=5144634012371033641_920_000_940_000 \
  b_clip_to_last_semantic_frame=False
```

生成背景 OCC：

```bash
python gt_generate/background_object_occ/generate_background_object_occ.py \
  data_path=$WAYMO_DATA_ROOT \
  sequence_name=5144634012371033641_920_000_940_000
```

可视化当前帧点云 + 真值：

```bash
python gt_generate/visualize_static_scene_gt.py 1
```

只可视化真值：

```bash
python gt_generate/visualize_static_scene_gt.py 2
```

训练地形预测模型：

```bash
python gt_generate/network_prediction/FH_train_terrain_model.py \
  data_path=$WAYMO_DATA_ROOT \
  sequence_name=5144634012371033641_920_000_940_000
```
