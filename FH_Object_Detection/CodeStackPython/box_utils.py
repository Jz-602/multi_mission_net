import numba
import numpy as np


def PointsInBoxes(points, boxes):
    indices = np.zeros((points.shape[0], boxes.shape[0]), dtype=bool)
    points_in_boxes_jit(points, boxes, indices)
    return indices


@numba.njit
def points_in_boxes_jit(points, boxes, indices):
    '''
    Input:
        points: float array [N, *],
        boxes:  float array[M, 7] or [M, 9],
                with first 6 dimensions x, y, z, length, width, height, last dimension yaw angle
    return:
        bool array of shape [N, M]
    '''
    num_points = points.shape[0]
    num_boxes = boxes.shape[0]
    for j in range(num_boxes):
        for i in range(num_points):
            if np.abs(points[i, 2] - boxes[j, 2]) <= boxes[j, 5] / 2.0:
                cosa = np.cos(boxes[j, -1])
                sina = np.sin(boxes[j, -1])
                shift_x = points[i, 0] - boxes[j, 0]
                shift_y = points[i, 1] - boxes[j, 1]
                local_x = shift_x * cosa + shift_y * sina
                local_y = -shift_x * sina + shift_y * cosa
                indices[i, j] = np.logical_and(np.abs(local_x) <= boxes[j, 3] / 2.0,
                                               np.abs(local_y) <= boxes[j, 4] / 2.0)

