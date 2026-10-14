import numpy as np
from CTools.CDataLoaderAX7 import CDataLoaderAX7
from CTools.CVisualizer import CVisualizer
vis = CVisualizer()

class CMapperAX7(CDataLoaderAX7):
    def mapper_with_RT_xyz(self,scan_files,Rs,ts,start_frame,end_frame,bool_vis):
        points_map = np.zeros([1, 3], dtype=np.float32)
        for i in range(start_frame,end_frame):
            print("%d / %d" % (i,end_frame))
            scan_name = scan_files[i]
            points_local = self.load_pcd_xyz_wo_nan(scan_name)
            points_local = points_local.transpose()
            R = Rs[i]
            t = ts[i]
            tt = np.tile(t, (points_local.shape[1], 1)).transpose()

            points_global = (R @ points_local + tt)
            points_global = points_global.transpose()
            points_map = np.vstack((points_map, points_global))
        points_map = points_map[1:,:]
        if bool_vis:
            vis.vis_cloud(points_map)
        return points_map








