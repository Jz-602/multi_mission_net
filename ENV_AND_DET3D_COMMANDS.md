# 环境变量与 det3d 编译命令备忘

本文档用于在本机和训练服务器之间同步代码后，快速设置机器相关路径，并重新编译 `det3d` 中的 CUDA / C++ 扩展。

## 1. 设置数据路径

每台机器只需要在 `torch_OD` 环境里设置一次自己的路径，不要把这些绝对路径写死进代码。

本机示例：

```bash
conda activate torch_OD

conda env config vars set WAYMO_DATA_ROOT=/media/estella/Osmanthus/aa_data/FH_data/FH_waymo_small

conda deactivate
conda activate torch_OD
```

服务器示例，把路径改成服务器实际位置：

```bash
conda activate torch_OD

conda env config vars set WAYMO_DATA_ROOT=/服务器上的数据路径

conda deactivate
conda activate torch_OD
```

Python 代码中建议这样读取：

```python
import os

waymo_data_root = os.environ["WAYMO_DATA_ROOT"]
```

## 2. 编译 ODSS_project/det3d/iou3d_nms

从项目根目录对应的实际路径进入 `ODSS_project/det3d/iou3d_nms` 后执行。

本机示例：

```bash
conda activate torch_OD

cd /media/estella/Osmanthus/aa_code/multi_mission/ODSS_project/det3d/iou3d_nms

rm -rf build
rm -f *.so
rm -f *.o

python setup.py build_ext --inplace
```

服务器示例：

```bash
conda activate torch_OD

cd /media/gk/data/lyf/aa_code/multi_mission/ODSS_project/det3d/iou3d_nms

rm -rf build
rm -f *.so
rm -f *.o

python setup.py build_ext --inplace
```

如果服务器上的项目目录仍然叫 `mulit_mission`，把上面路径里的 `multi_mission` 改成实际目录名即可。
