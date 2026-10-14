# 三任务训练与消融

## 当前生效安排（2026-10-07）

本轮使用独立补框检测真值和XYZT四通道，完整798/202序列、逐轮重新抽50%、固定5973验证及八卡配置不变。按 `server_xyzt_completed_` 配置执行单帧48 → 同单帧初始化时序48 → 等预算单帧续训48 → 随机初始化时序48；不做无根组。旧单帧/时序权重在新检测GT上另作完整验证。输出 `log/server_runs_xyzt_completed_20261007/`；最新状态、数据绑定、进程和验收见 `SERVER_PROGRESS.md` 顶部。下文2026-10-03及更早安排仅为历史，冲突时以本段和当前交接为准。


## 当前执行约束（2026-10-03 用户最新要求，覆盖下文冲突的历史安排）

- 优先级：完整数据动态50%单帧48轮 → 定向诊断后的15+1时序48轮 → 同初始化等预算单帧续训 → 必要根筛选消融。86序列仅保留探索结果，不再是前置条件；第五组已在完整44轮恢复点暂停。
- 原始划分798训练序列/202验证序列；完整候选池23688训练目标、固定5973验证目标。每轮以seed0+零基epoch无放回抽11844目标，然后八卡DDP分片、补齐4次重复，共1481更新/轮、71088更新/48轮。`train_tokens.json`表示完整候选池；逐轮清单、摘要和覆盖在`epoch_sampling/epoch_NNN.json`。
- 唯一正式配置为`configs/server_dynamic50_single.yaml`及同前缀temporal/single_continuation/必要no_root配置，清单`data_manifests/waymo_three_task_dynamic50.json`绑定全29661候选，输出`log/server_runs_dynamic50`。清单生成完并验收前不能启动正式训练。旧fixed86/full50配置只保留历史复现，旧自动等待入口已经停用。
- 正式实验全部GPU0..7，每卡batch1，原学习率不机械放大，统一48×48m、Z[-7,5.8]m、输入0.05m/输出0.2m，float32+TF32、增强全关闭。单帧48轮连预定完整验证争取约24h，确需可适当超时；旧72h默认预算作废。数据准备另计。
- 预检须完成上边界量化实际修复、代表性高候选反向、真实八卡训练/保存恢复、非零检测的代表性验证计时；既有通过项可复用。单帧正式从零，诊断/86权重不得用于其初始化。单帧期间并行定向排查时序检测退化，时序长训练前完成有针对性的诊断。
- 固定完整验证轮次5/10/15/20/25/30/35/40/41/42/43/44/45/46/47/48；5/25/48主动输出固定训练与验证PNG。完整指标/各任务raw loss/根删留/TensorBoard/权重与问题记录保留；数据、日志、权重不进Git，分界线标签不动。
- 多卡时序验证在读取任何历史帧前，从rank0统一同步模型缓冲区（含BN running statistics），再执行原有history/target路径；保持普通BN训练和SyncBN关闭，避免train→val首clip先使用各rank本地缓冲区。

服务器自主执行总说明见 [SERVER_CODEX_HANDOFF.md](SERVER_CODEX_HANDOFF.md)。

入口继续使用 `LYF_train_ODSS.py` 与 `LYF_train_ODSS_multi.py`。输出为目标检测、地形高度和静态三维 OCC；当前输入体素 0.05 米、输出 0.2 米，所有数据增强关闭。

## 环境与数据

在 `multi_mission` conda 环境运行，数据目录由 `WAYMO_DATA_ROOT` 指定。使用原三任务标签目录，不使用老师五层基础转换目录。当前索引是项目的69/17序列划分（984训练目标/520验证目标），并非Waymo官方验证集；论文中需按项目划分描述。

服务器的同名原始 infos 可能覆盖全量1000序列。先运行
`python 3mission_model/tools/prepare_server_infos.py`，按仓库参考清单恢复固定监督目标及这些序列的全部历史帧。
工具合并原训练/验证 infos 池，另存数据根目录下的 `three_task_20260930_infos_*`，原文件不变；相同输出可复用，不同内容拒绝覆盖。
服务器配置读取这组固定索引，`sample_fraction=1.0` 表示使用已经选好的984/520目标，不是扩大数据范围。
项目17个验证序列在当前服务器属于原训练池，因此时序验证也必须使用新生成的完整历史索引。
后续全量实验应另存配置、划分与数据清单，保留本阶段结果。

服务器可能只完成 `pseudo_label_2`。开始训练前先按 [交接说明第4节](SERVER_CODEX_HANDOFF.md#4-环境标签准备与磁盘预算) 核实空间，补齐地形和静态OCC标签并验收。生成器默认导出全部点云帧；地形四张图每帧约8.32MB，不能按1504个监督目标的空间估计直接启动全帧生成。现已支持 `--target-manifest`，可仅导出训练/验证监督目标，完整建图输入保留；相同目标与全帧导出的字节一致性已检查，命令见 [标签生成说明](../gt_generate/README.md)。

```bash
conda activate multi_mission
export WAYMO_DATA_ROOT=/media/gk/data/lyf/aa_data/FH_waymo
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

检测 CUDA 扩展须在每台机器按其 PyTorch/CUDA/GPU 重新编译；不要复制本机 `.so` 到服务器。在 `3mission_model/det3d/iou3d_nms` 内执行 `python setup.py build_ext --inplace`，`CUDA_HOME` 指向该环境匹配的 CUDA Toolkit。不要照搬本机 GPU 架构参数到服务器。

## 本机检查

从代码仓库根目录运行。`local_overfit.yaml` 固定一个训练目标帧，用于验证可学习性和输出图，不能作为验证集精度结果。

```bash
python 3mission_model/LYF_train_ODSS.py --config 3mission_model/configs/local_overfit.yaml --output 3mission_model/log/local_overfit
python 3mission_model/tools/visualize_three_task.py --config 3mission_model/configs/local_overfit.yaml --checkpoint 3mission_model/log/local_overfit/checkpoints/epoch_100.pth --index 10 --split train --output 3mission_model/log/local_overfit/comparison.png --loss-log 3mission_model/log/local_overfit/train_metrics.jsonl
```

时序检查按 15 个历史帧＋1 个目标帧运行，并从单帧权重初始化：

```bash
python 3mission_model/LYF_train_ODSS_multi.py --config 3mission_model/configs/local_temporal.yaml --init-checkpoint 3mission_model/log/local_overfit/checkpoints/epoch_100.pth --output 3mission_model/log/local_temporal --stop-after-epoch 4
python 3mission_model/LYF_train_ODSS_multi.py --config 3mission_model/configs/local_temporal.yaml --resume 3mission_model/log/local_temporal/checkpoints/latest.pth --output 3mission_model/log/local_temporal
```

需要时序预测图时，对 `tools/visualize_three_task.py` 使用时序配置与checkpoint并加 `--temporal`；该工具直接读取真实预测，标题明确标记train/val样本。

`--stop-after-epoch` 只限制本次运行轮数，学习率调度仍按配置的完整轮数构建。`--resume` 恢复模型、优化器、调度器、epoch 和 step；`--init-checkpoint` 只加载权重并开始新的训练阶段。两者互斥。按 epoch 边界恢复，不承诺逐 batch 随机状态完全复现。

资源检查（正式模型、实际标签、AdamW 前反向）：

```bash
python 3mission_model/tools/check_three_task.py --config 3mission_model/configs/server_single.yaml --index 10 --steps 2 --output 3mission_model/log/resource_single.json
python 3mission_model/tools/check_three_task.py --config 3mission_model/configs/server_temporal.yaml --temporal --index 10 --steps 2 --output 3mission_model/log/resource_temporal.json
```

首次迭代包含稀疏卷积内核选择开销；稳定速度取后续迭代，显存包含首次 AdamW 状态。时序报告同时包含历史构建和目标帧耗时。

## 服务器实验顺序

`server_single.yaml` / `server_temporal.yaml` 统一采用48×48米XY（±24米）、原Z范围、batch=1。服务器RTX4090 24GiB资源验收后，五组对照统一关闭OCC激活重算以减少重复计算；模型、任务损失及候选规则不变。扫描全部984训练目标的4070个合法时序clip后，最大1407936叶候选及最大融合x3样本均通过三次真实AdamW更新，包括无根筛选；最高分配13.10GiB、缓存14.25GiB，全部参数梯度有限。单帧最大候选稳定约0.50秒，时序高峰约1.25秒；首次内核选择不计入稳定步时。详情及限制见 `SERVER_PROGRESS.md`。

该范围最初基于12GB本机时序OOM共同收缩，服务器没有进一步缩小。原 `LYF_ODSS_config*.yaml` 保留160×160米参考范围，不要将其视作24GiB已验收配置。所有对照使用相同空间范围、数据划分、种子、batch和重算设置；后续50%大数据阶段仍须以实际峰值、吞吐和完整验证耗时确认排程。

```bash
python 3mission_model/LYF_train_ODSS.py --config 3mission_model/configs/server_single.yaml --output 3mission_model/log/server_single
python 3mission_model/LYF_train_ODSS_multi.py --config 3mission_model/configs/server_temporal.yaml --init-checkpoint 3mission_model/log/server_single/checkpoints/epoch_48.pth --output 3mission_model/log/server_temporal
```

主要消融配置：

| 对照 | 配置 | 初始化与比较 |
|---|---|---|
| 单帧＋根删留 | `server_single.yaml` | 单帧基础训练 |
| 时序＋根删留 | `server_temporal.yaml` | 从相同单帧模型初始化 |
| 单帧继续训练 | `server_single_continuation.yaml` | 同一单帧权重，补齐与时序相同的额外更新次数 |
| 单帧全部保留 | `server_single_no_root.yaml` | 独立训练，根头及其辅助损失均关闭 |
| 时序全部保留 | `server_temporal_no_root.yaml` | 从对应无根筛选单帧权重初始化 |

固定86序列的五组串行入口是 `tools/run_server_experiments.py`。主agent完成标签、空间和GPU资源验收后手动调用；脚本不进入后续全量阶段。可先只展示从头运行的命令（不读取标签或checkpoint）：

```bash
python 3mission_model/tools/run_server_experiments.py --dry-run --gpus 0
python 3mission_model/tools/run_server_experiments.py --gpus 0 --output-root 3mission_model/log/server_runs_20260930
```

顺序固定为单帧48轮、时序额外48轮、单帧额外48轮、无根单帧48轮、无根时序额外48轮。每组在5/25/48轮暂停，检查该轮验证记录，再串行输出train索引10及val索引0/100/300的三任务图和原始任务loss曲线。时序与单帧继续训练绑定同一个单帧`epoch_48.pth`，无根时序绑定无根单帧`epoch_48.pth`；主比较用最后48轮。`--gpus 0,1`可统一使用两卡训练/验证，出图只用首卡；同一次五组运行及恢复必须保持卡数一致。

恢复时重新调用相同`--output-root`：已有`latest.pth`必用`--resume`，已到达的停点不再训练，仅补缺失验证或图片。每条验证必须覆盖520帧且epoch不重复；最终历史必须恰好包含5/10/15/20/25/30/35及40至48轮，主比较读取历史中的48轮记录。默认目录为`3mission_model/log/server_runs_20260930`；各组在独立子目录下保存生产入口的`run_history.jsonl`，每条命令另存manifest/config副本、stdout及`commands.jsonl`起止/PID/退出码，并追加`SERVER_PROGRESS.md`。发现日志超出checkpoint、权重缺失、初始化不符或子命令失败立即停止，由主agent审计；不会清理日志、删权重或自动调整预算。

根筛选消融需重新训练。只在推理时关闭筛选属于同权重覆盖诊断。OCC 扩圈开关为 `model.occ.b_expand_roots`；更换它也需要独立配置与训练。第一版不开展额外结构搜索。

验证使用同一入口追加 `--validate-only` 和 `--resume <checkpoint>`。多卡使用 `torchrun --nproc_per_node=<卡数>` 启动相同入口。`sync_batchnorm` 始终为 false；同 GPU 多进程的条件分支测试不代替服务器多 GPU NCCL 首批检查。

每个输出目录保存完整合并配置、样本 token 列表、代码文件摘要、初始化权重内容校验值、逐步原始任务损失、验证指标与 checkpoint。服务器配置绑定 `data_manifests/waymo_three_task_20260930.json`，记录本机984个训练和520个验证目标帧的标签内容哈希；启动时核对目标 token，复制清单到输出目录。此检查不重新读取全部标签字节，迁移后应先重新生成清单，比较两个 split 的 `label_content_sha256`。地形导出器原始版本仍未知，内容哈希不补足来源证明。OCC 最终指标把未生成位置视作空，完整范围内的正真值漏检进入分母；根级误删与扩圈覆盖损失分别统计。

服务器迁移审查曾发现 `16102220208346880_1420_000_1440_000` 的旧逐帧地形仍为yaw-only：文件形状、时间戳和完成摘要均通过，但四矩阵内容不匹配参考。独立重建确认global不变，full-RPY的30个监督目标全部匹配参考后，才备份并替换120个地形文件；原始输入、frame_meta和OCC未改。文件齐全不能代替内容清单和几何验收，过程及失败证据见 `SERVER_PROGRESS.md`。

## 定向检查

```bash
python 3mission_model/tests/test_terrain_head.py
python 3mission_model/tests/test_occ_head.py --gpu
python 3mission_model/tests/test_three_task_temporal.py --gpu --ddp
python 3mission_model/tests/test_occ_checkpoint.py --gpu --ddp
python 3mission_model/tools/benchmark_occ_operator.py --gpu --output 3mission_model/log/occ_operator.json
python 3mission_model/tools/audit_three_task_labels.py --help
```

算子基准比较相同坐标、权重和梯度；小规模资源测试用于确认实现有效，不能推出整网效率优劣。地形重算审计明确检查的样本范围，不把一次精确匹配扩展为全量标签质量认证。

## 数据迁移核对

```bash
python 3mission_model/tools/build_label_manifest.py --config 3mission_model/configs/server_single.yaml --output 3mission_model/log/server_label_manifest.json
```

比较新清单与仓库清单中 train/val 的 `token_sha256` 和 `label_content_sha256`；内容变动应另存版本，不覆盖本次实验依据。服务器重新生成标签时先审计差异、记录导出版本，再将全部正式和消融配置的 `label_manifest` 统一绑定到验收后的新清单，目标划分保持一致。此工具仅读取文件并计算摘要，不修改标签。
