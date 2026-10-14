import open3d as o3d
import numpy as np

class CPointCloudProcesser():
    @staticmethod
    def extract_complement_from_two_clouds(cloud_big, cloud_small, min_search_dist):
        points_big = np.asarray(cloud_big.points, dtype=np.float32)
        points_small = np.asarray(cloud_small.points, dtype=np.float32)
        print("points_big:",points_big.shape)
        print("points_small:",points_small.shape)

        # 大点云构建kdtree
        # 小点云遍历每一个点
        big_kdtree = o3d.geometry.KDTreeFlann(cloud_big)
        ind_intersect = []
        for i in range(points_small.shape[0]):
            print("%d / %d" % (i, points_small.shape[0]))

            pointSel = points_small[i]
            [_, idx, dist] = big_kdtree.search_knn_vector_3d(pointSel, 1)
            dist = np.array(dist, dtype=np.float32)

            if dist < min_search_dist:
                idx_np = np.asarray(idx, dtype=np.int32)
                ind_intersect.append(idx_np)
        ind_intersect = np.asarray(ind_intersect, dtype=np.uint32).squeeze(1)
        ind_all = np.arange(0, points_big.shape[0])
        ind_complement = np.setdiff1d(ind_all, ind_intersect, assume_unique=False)
        return ind_intersect, ind_complement

if __name__ == '__main__':
    pc_processer = CPointCloudProcesser()
    cloud_big = o3d.io.read_point_cloud("/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/_annotation/map/dynamic_map_wo_label.pcd",remove_nan_points=True)
    cloud_small = o3d.io.read_point_cloud("/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/2022-01-12-16-48-31/LidarData/1641977318051.pcd",remove_nan_points=True)
    min_search_dist = 0.1
    ind_intersect, ind_complement = pc_processer.extract_complement_from_two_clouds(cloud_big, cloud_small, min_search_dist)
    print(1)