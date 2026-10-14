# AGENTS.md

## 项目目标

项目目标是基于 Waymo LiDAR 感知任务，逐步实现一个带有 memory 机制的统一多任务网络。整体上希望采用一套共享的 3D voxel / sparse-conv 主干，在 3D 空间中完成点云体素特征提取与编码，再分出面向不同任务的分支：一类保留 3D 特征用于语义分割等需要较强空间细节的任务，另一类投影到 BEV 表达，用于目标检测及后续可能的占据相关任务。这样可以把 3D 特征的几何细节能力和 BEV 表达在检测上的高效性结合起来，在共享主干的前提下完成多任务联合优化。（参考论文[多任务网络](/media/estella/Osmanthus/aa_code/multi_mission/多任务.pdf) ）

在此基础上，后续进一步引入 memory 机制，使历史帧信息能够通过位姿对齐后参与当前帧的 3D 特征编码或 BEV 特征融合。最终目标不是只做单帧的检测加分割拼接，而是形成一个同时统一 3D 主干、BEV 任务头和跨帧 memory 的感知框架，让当前帧预测既利用单帧几何信息，也能利用时序累积信息。

## 项目已有内容（必要时可以去查看）

- [FH_Object_Detection/2025-Waymo-OD](/media/estella/Osmanthus/aa_code/multi_mission/FH_Object_Detection/2025-Waymo-OD) 是老师已经跑过、效果验证较好的 Waymo 目标检测代码，可作为后续多任务检测分支的稳定基线。
- [FH_Object_Detection/2025-waymo-seg](/media/estella/Osmanthus/aa_code/multi_mission/FH_Object_Detection/2025-waymo-seg) 是基于 spconv 跑通的 MinkUNet 语义分割代码，目前训练可正常运行并观察到 loss 下降，但还没有完成完整训练和验证。
- 上面两部分目前已经共用一套 dataset 与 train 流程，这为后续整理统一多任务训练链提供了直接基础。
- [ODSS_project](/media/estella/Osmanthus/aa_code/multi_mission/ODSS_project) 是 ODSS 基础功能已经实现的主工作目录，包含目标检测与语义分割多任务网络的共享 backbone、任务分支以及统一训练流程。
- [MEMORY_waymo](/media/estella/Osmanthus/aa_code/multi_mission/MEMORY_waymo) 是利用历史帧信息的 Waymo 语义分割实验目录，可作为后续 memory 数据流、配置和 spconv 语义分割网络实现的参考。
- [ODSS_memory](/media/estella/Osmanthus/aa_code/multi_mission/ODSS_memory) 是利用历史帧信息的语义分割和目标检测小型多任务网络实验目录，可作为 memory 多任务结构与训练流程的主要参考。
- [PTV3](/media/estella/Osmanthus/aa_code/multi_mission/PTV3) 是当前新的语义分割模型基座实验目录，目前基础性能优于原 MinkUNet / spconv 语义分割基线，后续语义分割、时序融合和标签先验增强实验可优先在此推进。
- [Scene_Tag_Completion](/media/estella/Osmanthus/aa_code/multi_mission/Scene_Tag_Completion) 是 Waymo 全序列语义标签补全实验目录，目标是利用已有语义帧训练网络，为无语义真值帧推理整帧语义标签。当前重点是生成并使用传播语义先验，将网络输入由 `[N, 3] xyz` 扩展为 `[N, 4] xyz + propagated_label`；详细流程记录在目录内文档。这个项目的产出是每一帧的语义信息，mIoU 高达 98.5；如果后续有需要可以使用 `pseudo_label_2`。
- [gt_generate](/media/estella/Osmanthus/aa_code/multi_mission/gt_generate) 是 Waymo 真值构建目录，主要用于生成地面真值、静态场景真值和目标检测真值。
- 服务器上的项目目录名也使用 `multi_mission`，不要再使用旧拼写 `mulit_mission`。
- [lyf_tools.py](/media/estella/Osmanthus/aa_code/multi_mission/lyf_tools.py) 提供了一些常用工具，包括点云可视化 `visualize_point_cloud`、体素下采样 `voxel_downsample`、两族点云最近邻搜索 `nearest_points_cKDTree`、Waymo 位姿读取 `LoadPosesFromPoseTxt`，以及车体坐标系点云转世界坐标系点云的 `CarToWorld`。
- 本机默认 conda 环境为 `multi_mission`，服务器 conda 环境为 `multi_mission`。

## 数据说明

- 数据主目录通过 conda 环境变量 `WAYMO_DATA_ROOT` 设置，不要在代码或配置里写死机器相关路径。
- 服务器上的 Waymo 数据主目录是 `/media/gk/data/lyf/aa_data/FH_waymo`。
- 设置方式参考 [ENV_AND_DET3D_COMMANDS.md](/media/estella/Osmanthus/aa_code/multi_mission/ENV_AND_DET3D_COMMANDS.md)，核心命令是 `conda env config vars set WAYMO_DATA_ROOT=...`，设置后需要重新激活当前机器对应的 conda 环境。

## 项目小目标

1. 生成 Waymo 的场景补全真值，优先保证输出格式稳定、可批处理、可复查。

## 未来优化方向

- 训练时对相机位姿或外参施加小幅旋转、平移扰动，增强 view transformation 对标定误差的鲁棒性。


## 代码规范

严格遵循以下规则：

1. 类名使用 `PascalCase`
2. 函数名使用 `PascalCase`
3. 普通变量、成员变量、dataclass 字段使用 `snake_case`
4. `bool` 变量使用 `b_` 前缀
5. 模块级常量使用全大写下划线
6. 函数内或类中定义的常量可使用 `k_` 前缀
7. 常见缩写允许保留；若为大写缩写块，在函数名/类名中整体保持大写
8. Python 魔法方法保持原生写法
9. 能复用已有函数时优先扩展，不要复制近似实现。
10. 注释要贴近处理步骤本身；较长流程可用 `## 读取...`、`## 合并...`、`## 可视化...` 这类分段注释。
11. 涉及点云、位姿、检测框、语义标签时，必须说明输入数据的 shape、含义和所处坐标系；涉及坐标变换时，必须明确点、框、pose 各自在哪个坐标系以及变换方向。
12. 面向训练的输出优先保证格式稳定、可批处理、可复查。
13. numpy / 点云处理优先采用直接、可排查的批量化写法；若循环更直观且规模可接受，也可以保留。

## Git 规范

1. commit message 使用中文，简洁说明本次修改目的。
2. 一个完整修改目标对应一个 commit，避免一行一提交，也避免把无关修改混在同一个 commit。
3. 默认可以在完成并验证一个明确任务后执行本地 commit，但不自动 push 到远端。
4. 只有用户明确要求“上传”“同步到 GitHub”“push”或类似操作时，才执行远端 push。
5. 提交前必须检查 `git status`，确认没有误加入数据、权重、日志、编译产物或机器本地文件。

## 执行准则

**权衡说明**：本准则更偏向严谨性而非速度。对于简单琐碎的任务，可酌情灵活使用。

### 1. 编码前先思考

**不随意假设，不掩盖困惑，明确权衡方案。**

实现之前：

- 清晰说明你的假设。如有不确定，主动提问。
- 若存在多种理解方式，应全部列出，不暗自选定。
- 若有更简洁的方案，主动提出；合理时可给出不同建议。
- 若内容不清晰，立即停止。明确指出困惑点并提问。

### 2. 简洁优先

**用最少代码解决问题，不做任何预判性设计。**

- 不实现需求以外的任何功能。
- 单次使用的代码不做抽象封装。
- 不添加未被要求的“灵活性”或“可配置性”。
- 不为不可能出现的场景编写异常处理。
- 若本可 50 行实现却写了 200 行，应重写。自我检验：资深工程师是否会认为这段代码过度复杂？如果是，就简化。

### 3. 精准修改

**只修改必需内容，只清理自身产生的冗余代码。**

修改现有代码时：

- 不“优化”相邻代码、注释或格式。
- 不重构没有问题的逻辑。
- 遵循项目现有风格，即便你有不同写法。
- 发现无关的死代码，只需提示，不擅自删除。

当你的修改产生孤儿代码时：

- 删除因**你的改动**而失效的导入、变量、函数。
- 除非明确要求，否则不删除原本就存在的死代码。

检验标准：每一行修改都应能直接追溯到用户的需求。

### 4. 目标驱动执行

**定义成功标准，循环验证直至通过。**

将任务转化为可验证目标：

- “添加校验”→“为非法输入编写测试用例并使其通过”
- “修复 Bug”→“编写可复现问题的测试用例并使其通过”
- “重构 X”→“确保重构前后测试均能正常通过”

多步骤任务需给出简要计划：

```text
1. [步骤] → 验证：[检查项]
2. [步骤] → 验证：[检查项]
3. [步骤] → 验证：[检查项]
```

清晰的成功标准可让模型自主迭代完成；模糊标准（如“让它正常运行”）则需要反复沟通确认。

**准则生效的判断标准**：代码差异（diff）中无关改动减少、因过度复杂导致的重写减少、疑问在实现前提出，而非出错后补救。
