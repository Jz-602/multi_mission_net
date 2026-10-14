

def YPR2RotationMatrix(ypr):
    # our coordinate: x->right->pitch, y->forward->roll, z->up->yaw
    Y, P, R = ypr[0], ypr[1], ypr[2]

    crz, srz = math.cos(Y), math.sin(Y)
    crx, srx = math.cos(P), math.sin(P)
    cry, sry = math.cos(R), math.sin(R)

    R_y = np.array([[cry, 0, sry],
                    [0, 1, 0],
                    [-sry, 0, cry]
                    ])

    R_x = np.array([[1, 0, 0],
                    [0, crx, -srx],
                    [0, srx, crx]
                    ])

    R_z = np.array([[crz, -srz, 0],
                    [srz, crz, 0],
                    [0, 0, 1]
                    ])

    R_w_l = R_z @ R_x @ R_y
    return R_w_l

def R2Ang_3D_Our(R):
    # Z1X2Y3, applicable for our coordinate (X->right, Y->forward, Z->upward)
    #    angle[0] = yaw; angle[1] = pitch; angle[2] = roll
    angle = np.empty((3,1))
    angle[1] = math.asin(R[2,1]);
    cp = math.cos(angle[1]);
    angle[2] = -math.atan2((R[2,0]/cp),(R[2,2]/cp));
    angle[0] = -math.atan2((R[0,1]/cp),(R[1,1]/cp));
    return angle

def Rot_z(angle):
    crz, srz = math.cos(angle), math.sin(angle)
    R_z = np.array([[crz, -srz, 0],
                    [srz, crz, 0],
                    [0, 0, 1]
                    ])
    return R_z




class Box:
    def __init__(self, center, size, orientation):
        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation

    def RotationMatrix(self):
        return self.orientation.rotation_matrix

    def Translate(self, x):
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        :return: <None>.
        """
        self.center += x

    def Rotate(self, quaternion: Quaternion):
        """
        Rotates box.
        :param quaternion: <Quaternion>. Rotation to apply.
        :return: <None>.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation


    def Corners(self, wlh_factor: float=1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        # x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        # y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        # z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])

        # 3D bounding box corners. (Ours: x points right, y forward, z up.)
        x_corners = w / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        y_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])

        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners





def CreateO3dboxFromBoxCorners(corners, color=None):
	'''
	corners: 8 corners(x, y, z)
	corners: array = 8*3
    # x->forward, y->left, z->up
	#         0 -------- 1
	#        /|         /|
	#       4 -------- 5 .
	#       | |        | |
	#       . 3 -------- 2
	#       |/         |/
	#       7 -------- 6
	'''
	lines = [[0, 1], [1, 2], [2, 3], [3, 0],
				[4, 5], [5, 6], [6, 7], [7, 4],
				[0, 4], [1, 5], [2, 6], [3, 7],
             [5, 7], [4, 6]]
	if color == None:
		color = [1, 0, 0] # red
	colors = [color for i in range(len(lines))]
	line_set = o3d.geometry.LineSet()
	line_set.points = o3d.utility.Vector3dVector(corners)
	line_set.lines = o3d.utility.Vector2iVector(lines)
	line_set.colors = o3d.utility.Vector3dVector(colors)

	return line_set


def VisualizePCandBB(PC, BB):
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window()
    vis2.get_render_option().point_size = 2  # set points size

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis2.add_geometry(mesh_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(PC.points.T)
    vis2.add_geometry(pcd)

    o3d_box = CreateO3dboxFromBoxCorners(BB.Corners().T)
    vis2.add_geometry(o3d_box)

    vis2.run()




def VisualizePCandBBin2D(PC, BB):
    if 0:
        plt.figure()
        plt.axis([-30,30,-30,30])
        plt.scatter(PC.points[0,:], PC.points[1,:])

        # plt.Rectangle(xy=(0.2, 0.3), width=1.2, height=0.8, color='r',
        #               fill=False, linewidth=9)

        rect1 = plt.Rectangle(xy=(BB.center[0], BB.center[1]), width=BB.wlh[0], height=BB.wlh[1], color='r',
                      fill=False, linewidth=6)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(rect1)

        plt.axis('equal')
        plt.show()
    else:

        bever = CBEVer(PC.points.T, bound=[-30, 30, -30, 30, -2.73, 1.27])
        bever.extract_inlier()
        height_map = bever.xyz_to_bev()

        colored_height_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype='uint8')
        colored_height_map[:,:,0] = height_map
        colored_height_map[:,:,1] = height_map
        colored_height_map[:,:,2] = height_map


        u,v = bever.XY2UV(BB.center[0], BB.center[1])

        angle = R2Ang_3D_Our(BB.RotationMatrix())
        yaw = angle[0]


        rect = ((u, v), (BB.wlh[0]/bever.resolution, BB.wlh[1]/bever.resolution), int(-yaw*180.0/math.pi))   # in rotatedrect, angle is positive clockwise
        box = cv2.boxPoints(rect)  # cv2.cv.BoxPoints(rect) for OpenCV <3.x
        box = np.int0(box)
        cv2.drawContours(colored_height_map, [box], 0, (0, 0, 255), 2)

        cv2.imshow("bev_map", colored_height_map)
        print(yaw)
        print(int(-yaw*180.0/math.pi))
        cv2.waitKey(-1)





def CropPC(PC, box, offset=0, scale=1.0):

    maxi = np.max(box.Corners(scale), 1) + offset
    mini = np.min(box.Corners(scale), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    return new_PC


def CropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):

    new_PC = CropPC(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.RotationMatrix())
    trans = -new_box.center

    # align data
    new_PC.Translate(trans)
    new_box.Translate(trans)
    new_PC.Rotate((rot_mat))
    new_box.Rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = CropPC(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC




def GenerateBoxesFromSearchSpace(box, search_space=[[0, 0, 0]]):
    # Geenrate more candidate boxes based on prior and search space
    # Input : Prior position, search space and search size
    # Output : List of boxes

    candidate_boxes = []

    for offset in search_space:

        rot_quat = box.orientation
        trans = np.array(box.center)

        new_box = copy.deepcopy(box)

        new_box.Translate(-trans)
        new_box.Rotate(rot_quat.inverse)

        # REMOVE TRANSfORM
        if len(offset) == 3:
            new_box.Rotate(
                Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))
        new_box.Translate(np.array([offset[0], offset[1], 0]))

        # APPLY PREVIOUS TRANSFORMATION
        new_box.Rotate(rot_quat)
        new_box.Translate(trans)

        candidate_boxes.append(new_box)

    return candidate_boxes


