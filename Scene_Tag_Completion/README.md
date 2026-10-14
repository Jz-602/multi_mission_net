# Scene_Tag_Completion

## 实验目标

`Scene_Tag_Completion` 用于补全 Waymo 全序列语义标签。Waymo 原始语义标注只覆盖部分关键帧，本实验希望利用已有语义真值帧训练网络，为同一序列中没有语义真值的点云帧推理整帧语义标签。

当前核心思路是给点云输入增加一维传播语义先验：网络原始输入为 `[N, 3] xyz`，其中 `xyz` 是当前帧 LiDAR/车体坐标系下的点坐标；增强后输入为 `[N, 4] xyz + propagated_label`，第 4 维不是激光强度，而是从已有语义真值帧传播来的训练标签空间语义先验。没有可靠传播结果的点使用 `0`，表示无先验信息。

## 数据与坐标系约定

- 原始点云来自每个 Waymo sequence 的 `output/pcd/*.pcd`，读取后为 `points_xyz_lidar: [N, 3]`，坐标系是当前帧 LiDAR/车体坐标系。
- 原始语义真值来自 `output/SemanticSegmentation/*.npy`，当前约定为 shape `[N, 2]`，第二列是原始 Waymo 语义标签。
- 位姿来自 `output/pose_slam_euler.txt`，用于将当前帧 LiDAR/车体坐标系点云变换到世界坐标系。
- 传播语义先验保存到 `output/sequence_SemanticPrior/*.label`，每个 `.label` 是 `uint16` 一维数组，shape `[N]`，点顺序与对应 `.pcd` 一致，标签已经映射到训练标签空间。
- 训练标签映射由 `label_mapping/waymo_merge_example.yaml` 和配置文件中的 `common_variables.label_mapping` 控制，`0` 作为 unknown / ignore 类别。

## 标签传播流程

标签传播主要由 `GenerateFourLABELSemanticLabelsForWaymo.py` 生成，流程如下：

1. 扫描 `WAYMO_DATA_ROOT` 下所有可处理的 `sequence/output` 目录。目录需要同时包含 `pcd/`、`SemanticSegmentation/` 和 `pose_slam_euler.txt`。
2. 对同一 sequence 中已有语义真值的帧，读取当前帧点云 `points_xyz_lidar: [N, 3]` 和语义真值 `semantic_labels: [N]`。
3. 使用 Waymo pose 将这些有真值帧的点云从当前帧 LiDAR/车体坐标系变换到世界坐标系，得到 `world_points_xyz: [N, 3]`。
4. 将原始 Waymo 语义标签映射到训练标签空间，并过滤掉标签为 `0` 的 unknown 点。
5. 把所有有语义真值帧的有效点拼成稀疏语义地图，并构建 `cKDTree`。
6. 对 sequence 中每一帧点云，同样用当前帧 pose 变换到世界坐标系，在稀疏语义地图中查询最近邻语义点。
7. 若最近邻距离不超过 `PROPAGATE_RADIUS_METERS`，当前默认为 `0.5m`，则把最近邻语义标签作为该点的传播先验；否则该点先验标签置为 `0`。
8. 传播结果按帧保存为 `sequence_SemanticPrior/{frame_id}.label`，供训练和推理直接读取。

脚本中还会在传播失败的 unknown 点上尝试使用当前帧检测框补充一部分动态目标先验；这部分只影响传播先验，不替代最终语义监督标签。

## 防止训练标签泄漏

训练时使用有语义真值的帧作为监督目标，输出仍是整帧点云的语义标签。为了让网络适应“只有部分点有传播先验”的输入，并避免直接把当前帧真值泄漏进输入，生成有真值帧的传播先验时必须使用 leave-one-frame-out 方式：

- 如果当前帧没有语义真值，则使用同一 sequence 内全部有语义真值帧构建稀疏语义地图。
- 如果当前帧本身有语义真值，则构建传播地图时排除当前帧，只利用其它语义真值帧传播先验。

这样训练输入中的第 4 维 `propagated_label` 只表示其它帧传播来的零散语义先验，监督目标仍来自当前帧自己的完整语义真值。

## 训练与配置

当前目录保留了两类配置：

- `LYF_segmentation_xyz.yaml`：基础三维输入实验，`model.in_channels: 3`，输入为 `[N, 3] xyz`。
- `LYF_segmentation_xyzl.yaml`：语义先验增强实验，`model.in_channels: 4`，数据读取流程包含 `load_semantic_prior`，输入为 `[N, 4] xyz + propagated_label`。
- `LYF_segmentation_multi.yaml`：多帧拼接输入实验，`model.in_channels: 4`，数据读取流程包含 `load_pseudo_label`，输入为多帧对齐后的 `[sum_N, 4] xyz + pseudo_label_1`。

`WaymoDatasetV2.py` 中的 `load_semantic_prior` 会从 `sequence_SemanticPrior` 读取 `.label` 文件，并把传播标签拼到点特征第 4 维。`semantic_prior_dropout` 可随机把一部分传播先验置为 `0`，用于降低网络对先验标签的过度依赖。

## 多帧拼接阶段

多帧阶段用于在 `pseudo_label_1` 基础上生成最终全序列语义真值。`pseudo_label_1` 由 `infer_pcd/infer_first_stage_pseudo_label.py` 推理得到，默认保存到每个 sequence 的 `output/label/pseudo_label_1/*.label`，每个文件是 `uint32`，shape `[N]`，与对应 PCD 点顺序一致。

第二阶段伪标签 `pseudo_label_2` 由 `infer_pcd/infer_second_stage_pseudo_label.py` 推理得到，默认使用 `checkpoints_multi/epoch_48.pth` 的多帧模型。推理时脚本读取 target frame 及其上下文帧的点云，将上下文帧通过 pose 对齐到 target frame LiDAR/车体坐标系，并把每帧的 `pseudo_label_1` 作为第 4 维输入特征。模型输出后只保存 target frame 原始点顺序对应的语义标签，默认路径为 `output/label/pseudo_label_2/*.label`，文件格式同样是 `uint32`、shape `[N]`。因此当前全序列语义补全链路为 `sequence_SemanticPrior -> pseudo_label_1 -> pseudo_label_2`。

`LYF_train_minkunet_seg_multi.py` 使用 `WaymoDataset_multi.py` 读取有人工语义真值的帧作为 target frame。配置中的 `input_frame_count` 控制输入总帧数，例如 `7`；`frame_interval` 控制相邻输入帧的间隔，例如 `2` 表示每两帧取一帧。若 target frame 是第 100 帧，且 target 在 7 帧窗口中的 offset 为 2，则输入帧为 `96, 98, 100, 102, 104, 106, 108`。

每个输入帧原始点云都是自身 LiDAR/车体坐标系下的 `points_xyz_lidar: [N, 3]`。拼接前，dataset 使用每帧 pose 将上下文帧点云先变换到世界坐标系，再变换回 target frame LiDAR/车体坐标系；第 4 维 `pseudo_label_1` 只随点拼接，不参与坐标变换。最终模型输入为 `[sum_N, 4]`。

训练 loss 只监督 target frame：target frame 点使用人工语义真值，其它上下文帧点的 `semantic_label` 填 `ignore_index=0`。为了适应序列首尾推理时只能使用历史帧或未来帧的情况，训练集会随机选择 target frame 在多帧窗口中的位置；验证集默认使用配置中的固定 `supervised_window_offset`。

## 推理目标

推理阶段输入为任意 Waymo 帧的 `points_xyz_lidar: [N, 3]` 以及对应传播先验 `prior_labels: [N]`，组合成 `[N, 4] xyz + propagated_label`。网络输出为整帧点云的语义标签，用于补全原始 Waymo 中没有语义真值的帧。

## 关键文件

- `GenerateFourLABELSemanticLabelsForWaymo.py`：生成并可视化 `sequence_SemanticPrior/*.label`。
- `WaymoDatasetV2.py`：读取点云、语义真值和传播语义先验，并组织 `[N, 3]` 或 `[N, 4]` 输入。
- `LYF_train_minkunet_seg_xyz.py`：三维输入基线训练入口。
- `LYF_train_minkunet_seg_xyzl.py`：四维语义先验输入训练入口。
- `LYF_train_minkunet_seg_multi.py`：多帧拼接输入、单帧监督训练入口。
- `infer_pcd/infer_first_stage_pseudo_label.py`：使用单帧 xyzl 模型生成 `pseudo_label_1`。
- `infer_pcd/infer_second_stage_pseudo_label.py`：使用多帧模型生成 `pseudo_label_2`。
- `LYF_segmentation_xyz.yaml`：三维输入配置。
- `LYF_segmentation_xyzl.yaml`：四维语义先验输入配置。
- `LYF_segmentation_multi.yaml`：多帧拼接输入配置。
