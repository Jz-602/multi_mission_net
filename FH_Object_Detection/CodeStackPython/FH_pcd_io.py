

import open3d as o3d
import numpy as np
import os


def FH_write_pcd_binary_xyzI(points, filename):
    """
    将点云保存为二进制 PCD 文件，格式：pointXYZI（x, y, z, intensity），float32 小端。

    参数
    ----
    filename : str
        输出 pcd 文件路径（会被覆盖）。
    points : np.ndarray, shape (N, 3) or (N, 4)
        点云数据，列为 [x, y, z] 或 [x, y, z, intensity]。
        如果只给了 3 列，会自动把 intensity 列补为 0.

    备注
    ----
    - 输出的 PCD header 使用 VERSION .7，DATA binary。
    - 使用 float32 小端格式保证与 PCL/其他可视化器的兼容性。
    """
    # 验证输入
    if not isinstance(points, np.ndarray):
        raise TypeError("points 必须为 numpy.ndarray")
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("points 必须为形状 (N,3) 或 (N,4) 的数组")

    # 准备 Nx4 数组（x,y,z,intensity）
    if points.shape[1] == 3:
        n = points.shape[0]
        pts = np.zeros((n, 4), dtype=np.float32)
        pts[:, :3] = points.astype(np.float32)
        # intensity 保持为 0
    else:
        pts = points.astype(np.float32).copy()

    N = pts.shape[0]

    # 确保使用小端 float32（'<f4'）
    pts_le = pts.astype('<f4')

    # 构造 header
    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION .7",
        "FIELDS x y z intensity",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        f"WIDTH {N}",
        "HEIGHT 1",
        f"POINTS {N}",
        "DATA binary"
    ]
    header = "\n".join(header_lines) + "\n"

    # 写入文件（先 ASCII header，再 binary data）
    # 覆盖同名文件
    with open(filename, "wb") as f:
        f.write(header.encode("ascii"))
        # numpy.ndarray.tofile 可直接写入二进制数据
        pts_le.tofile(f)

    # 可选：返回文件大小以便检查
    return os.path.getsize(filename)

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
