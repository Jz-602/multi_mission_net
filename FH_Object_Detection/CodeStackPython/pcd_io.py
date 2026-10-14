

import open3d as o3d
import numpy as np
import os

def FH_ReadBinaryXYZIPointCloud(pcd_path):

    header_lines = []
    data_start = 0
    point_size = 0
    fields = []
    sizes = []

    with open(pcd_path, 'rb') as f:
        while True:
            line = f.readline()
            try:
                decoded_line = line.decode('utf-8').strip()
                header_lines.append(decoded_line)

                if decoded_line.startswith('FIELDS'):
                    fields = decoded_line.split()[1:]
                elif decoded_line.startswith('SIZE'):
                    sizes = list(map(int, decoded_line.split()[1:]))
                elif decoded_line.startswith('DATA'):
                    data_format = decoded_line.split()[1]
                    if data_format != 'binary':
                        raise ValueError("非二进制格式")
                    data_start = f.tell()
                    break

            except UnicodeDecodeError:
                # 二进制数据开始，跳出循环
                data_start = f.tell() - len(line)
                break

    # 计算点结构
    point_size = sum(sizes)
    file_size = os.path.getsize(pcd_path)
    num_points = (file_size - data_start) // point_size

    # 直接映射到numpy数组
    with open(pcd_path, 'rb') as f:
        f.seek(data_start)
        binary_data = np.fromfile(f, dtype=np.uint8, count=num_points * point_size)

    # 重新解释为浮点数数组
    points_array = binary_data.view(np.float32).reshape(num_points, -1)

    # 提取XYZ坐标
    xyz_indices = [fields.index(field) for field in ['x', 'y', 'z']]
    xyz = points_array[:, xyz_indices]

    # 提取强度信息
    if 'intensity' in fields:
        intensity_idx = fields.index('intensity')
        intensity = points_array[:, intensity_idx]


    return xyz, intensity


def FH_ReadASCIIXYZIPointCloud(filename):
    # 1. 使用Open3D读取点云（获取XYZ坐标）
    pcd = o3d.io.read_point_cloud(filename)
    points_xyz = np.asarray(pcd.points)  # 获取XYZ坐标

    # 2. 手动解析PCD文件以提取强度值
    intensity_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()

        # 找到DATA行的位置
        data_start_idx = None
        for idx, line in enumerate(lines):
            if line.startswith('DATA'):
                data_start_idx = idx + 1
                break

        # 从DATA行开始读取数据
        if data_start_idx is not None:
            for line in lines[data_start_idx:]:
                # 假设数据格式是 x y z intensity ...
                data = line.strip().split()
                if len(data) >= 4:  # 确保有强度值
                    intensity = float(data[3])
                    intensity_list.append(intensity)

    # 将强度值归一化并映射到RGB颜色（灰度）
    intensity_array = np.array(intensity_list)
    if intensity_array.size > 0:
        # 归一化到 [0, 1] 范围
        intensity_normalized = (intensity_array - intensity_array.min()) / (intensity_array.max() - intensity_array.min())
        # 创建灰度颜色 (R=G=B=强度值)
        gray_colors = np.repeat(intensity_normalized[:, np.newaxis], 3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(gray_colors)

    # 可视化
    o3d.visualization.draw_geometries([pcd])
    return pcd
