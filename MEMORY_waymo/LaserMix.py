import numpy as np


def Deg2Rad(degree):
    return degree * np.pi / 180.0


def SplitByInclination(points, labels, bounds):
    ## points: [N, C]，当前帧 LiDAR/车体坐标系点特征，前三维为 xyz；labels: [N]，与 points 一一对应。
    rho = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    inclination = np.arctan2(points[:, 2], rho)

    point_parts = []
    label_parts = []
    upper_bound = None
    for bound in bounds:
        if upper_bound is None:
            mask = inclination > Deg2Rad(bound)
        else:
            mask = np.logical_and(inclination <= Deg2Rad(upper_bound), inclination > Deg2Rad(bound))
        point_parts.append(points[mask])
        label_parts.append(labels[mask])
        upper_bound = bound

    mask = inclination <= Deg2Rad(bounds[-1])
    point_parts.append(points[mask])
    label_parts.append(labels[mask])
    return point_parts, label_parts


def LaserMixAug(points, labels, points_another, labels_another):
    ## 输入点均在各自当前帧 LiDAR/车体坐标系；按垂直角分束后交替取两帧点，输出仍为 [M, 3] xyz 与 [M] label。
    labels = labels.reshape(-1)
    labels_another = labels_another.reshape(-1)

    strategy_bounds_list = [
        [-6.7, -13.4],
        [-5.0, -10.0, -15.0],
        [-4.0, -8.0, -12.0, -16.0],
        [-3.3, -6.6, -9.9, -13.2, -16.5],
    ]
    bounds = strategy_bounds_list[np.random.randint(len(strategy_bounds_list))]

    point_parts, label_parts = SplitByInclination(points, labels, bounds)
    point_parts_another, label_parts_another = SplitByInclination(points_another, labels_another, bounds)

    mixed_points = []
    mixed_labels = []
    for part_idx in range(len(point_parts)):
        if part_idx % 2 == 0:
            mixed_points.append(point_parts[part_idx])
            mixed_labels.append(label_parts[part_idx])
        else:
            mixed_points.append(point_parts_another[part_idx])
            mixed_labels.append(label_parts_another[part_idx])

    return (
        np.concatenate(mixed_points, axis=0).astype(np.float32),
        np.concatenate(mixed_labels, axis=0).astype(np.int32),
    )
