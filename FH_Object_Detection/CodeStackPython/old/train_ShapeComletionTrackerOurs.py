import os
import time
import logging


from pyquaternion import Quaternion

from tqdm import tqdm


from datetime import datetime

import torch
import torch.cuda
import torch.optim
import torch.utils.data

from NN_ShapeCompletion import Model

import open3d as o3d
import copy

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pylab as pylab

import math
import struct

import numpy as np


from PCLosses import ChamferLoss



# for reproducibility
torch.manual_seed(0)
np.random.seed(0)




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
    def __init__(self, center, size, orientation, label=np.nan):

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label

        self.rotation_matrix = self.orientation.rotation_matrix

        # self.corners = self.Corners()


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

class PointCloud:

    def __init__(self, points):
        """
        Class for manipulating and viewing point clouds.
        :param points: <np.float: 4, n>. Input point cloud matrix.
        """
        self.points = points
        if self.points.shape[0] > 3:
            self.points = self.points[0:3, :]

    def NumOfPoints(self):
        return self.points.shape[1]

    def Translate(self, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        :return: <None>.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def Rotate(self, rot_matrix):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        :return: <None>.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def Normalize(self, wlh):
        normalizer = [wlh[1], wlh[0], wlh[2]]
        self.points = self.points / np.atleast_2d(normalizer).T

class OurRS128TrackerTrainingDataset():

    def __init__(self, LiDARDataPath_):
        self.LiDARDataPath = LiDARDataPath_

        self.map_center = np.loadtxt(os.path.join(self.LiDARDataPath, 'Localizer/map_center.txt'), dtype=np.float64)
        data = np.loadtxt(os.path.join(self.LiDARDataPath, 'calculated_slam_all_pose.txt'), dtype=np.float64).reshape(-1, 7)
        self.allframe_timestamp_DB = data[:,0].astype(int)
        self.allframe_pose_DB = data[:,1:]

        param_file = '/home/hao/work/2023-CodeStack-DATMO/bin/parameters/DATMOTraining.ini'
        with open(param_file, 'r') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) == 0:
                    continue
                # elif values[0] == 'start_frame':
                #     self.start_frame = values[1]
                elif values[0] == 'static_vehicle_global_pose':
                    static_vehicle_global_pose = np.array(values[1:], dtype=np.float32)    # [x,y,z,yaw,pitch,roll]
                elif values[0] == 'static_vehicle_wlh':
                    static_vehicle_wlh = np.array(values[1:], dtype=np.float32)    # [width, length, height]


        self.R_w_v = YPR2RotationMatrix(static_vehicle_global_pose[3:] * math.pi / 180.0)
        self.t_w_v = static_vehicle_global_pose[0:3]
        self.wlh = static_vehicle_wlh
        # box_orientation = Quaternion(matrix = self.R_w_v)
        # self.global_box = Box(self.t_w_v, self.wlh, box_orientation)


    def NumOfFrames(self):
        return len(self.allframe_timestamp_DB)

    def __getitem__(self, index):
        if index>=len(self.allframe_timestamp_DB) or index<0:
            return False

        lidar_filename = os.path.join(self.LiDARDataPath, 'LidarData', f'{self.allframe_timestamp_DB[index]}.pcd')
        lidar_pcd = o3d.io.read_point_cloud(lidar_filename)

        points = np.asarray(lidar_pcd.points, dtype=np.float32).T

        lidar_xyzypr = self.allframe_pose_DB[index, :]
        R_w_l = YPR2RotationMatrix(lidar_xyzypr[3:])
        t_w_l = lidar_xyzypr[0:3] - self.map_center
        t_w_l = t_w_l.T

        R_l_w = R_w_l.T
        t_l_w = -R_l_w @ t_w_l

        R_l_v = R_l_w @ self.R_w_v
        t_l_v = R_l_w @ self.t_w_v + t_l_w;

        BB = Box(t_l_v, self.wlh, Quaternion(matrix = R_l_v))

        return PointCloud(points), BB

class VechicleModelDatabase:
    def __init__(self, model_filename):
        file_size = int(os.path.getsize(model_filename)/4)
        with open(model_filename, 'rb') as file:
            data = struct.unpack('f'*file_size, file.read(file_size*4))
            idx = 0
            self.range_list = []
            self.yaw_list = []
            self.pts_list = []
            while idx<len(data):
                range = data[idx]
                idx += 1
                yaw = data[idx]
                idx += 1
                num_of_pts = int(data[idx])
                idx += 1
                pts = np.array(data[idx:idx+num_of_pts*3])
                pts = pts.reshape(num_of_pts,3,order='F').T   # F: Fortran, same as MATLAB, column major
                idx += num_of_pts*3

                self.range_list.append(range)
                self.yaw_list.append(yaw)
                self.pts_list.append(pts)

        self.unique_range = np.asarray(list(set(self.range_list)))
        self.unique_yaw = np.asarray(list(set(self.yaw_list)))

    def GetModel(self, t_l_v, R_l_v):       # vehicle coord (box coord) to LiDAR coord
        idx = self.GetIdx(t_l_v, R_l_v)
        return self.pts_list[idx]



    def GetModelCode(self, t_l_v, R_l_v):
        idx = self.GetIdx(t_l_v, R_l_v)
        return self.model_code_all[:,idx]

    def GetIdx(self, t_l_v, R_l_v):
        xy_range = math.sqrt(t_l_v[0]*t_l_v[0] + t_l_v[1]*t_l_v[1])
        idx = np.abs(self.unique_range - xy_range).argmin()
        nearest_range = self.unique_range[idx]

        alpha = math.atan2(t_l_v[0], t_l_v[1]);
        R_ln_lo = Rot_z(alpha); # LiDAR-old coord to LiDAR-new coord
        R_ln_v = R_ln_lo @ R_l_v;
        angle = R2Ang_3D_Our(R_ln_v)
        yaw = angle[0] if angle[0] >= 0 else angle[0] + 2*math.pi
        idx = np.abs(self.unique_yaw - yaw).argmin()
        nearest_yaw = self.unique_yaw[idx]

        bool_array = np.logical_and(self.range_list == nearest_range, self.yaw_list == nearest_yaw)
        idx = np.squeeze(np.where(bool_array)[0])
        return idx

    def ComputeLatentCode(self, NN_model):
        self.model_code_all = np.empty((NN_model.bneck_size, len(self.pts_list)))
        for i in range(len(self.pts_list)):
            PC = PointCloud(self.pts_list[i])
            regularized_points = RegularizePC(PC, NN_model).cuda()
            model_code = NN_model.AE.encode(regularized_points)
            self.model_code_all[:,i] = np.squeeze(model_code.detach().cpu().numpy().T)

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

    rot_mat = np.transpose(new_box.rotation_matrix)
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

def GetOffsetBB(box, offset):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
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
    return new_box

def RegularizePC(PC, model):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != model.input_size:
            new_pts_idx = np.random.randint(
                low=0, high=PC.shape[1], size=model.input_size, dtype=np.int64)
            PC = PC[:, new_pts_idx]
        PC = PC.reshape(1, 3, model.input_size)

    else:
        PC = np.zeros((1, 3, model.input_size))

    return torch.from_numpy(PC).float()

def GetScoreGaussian(offset, sigma=1):
    coeffs = [1, 1, 1 / 5]
    dist = np.linalg.norm(np.multiply(offset, coeffs))
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return torch.tensor([score])


class KalmanFiltering():
    def __init__(self, bnd=[1, 1, 10]):
        self.bnd = bnd
        self.Reset()

    def Sample(self, n=10):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def AddData(self, data, score):
        score = score.clip(min=1e-5)  # prevent sum=0 in case of bad scores
        self.data = np.concatenate((self.data, data))
        self.score = np.concatenate((self.score, score))
        self.mean = np.average(self.data, weights=self.score, axis=0)
        self.cov = np.cov(self.data.T, ddof=0, aweights=self.score)

    def Reset(self):
        self.mean = np.zeros(len(self.bnd))
        self.cov = np.diag(self.bnd)
        if len(self.bnd) == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.array([])

class SiamseTrainDataset():
    def __init__(self,
                 model,
                 lidar_data_path,
                 vehicle_model_path,
                 regress="GAUSSIAN",
                 sigma_Gaussian=1,
                 offset_BB=0,
                 scale_BB=1.0):

        self.model_database = VechicleModelDatabase(vehicle_model_path)
        our_dataset = OurRS128TrackerTrainingDataset(lidar_data_path)

        self.model = model

        self.num_of_frames = len(our_dataset.allframe_timestamp_DB) - 2600

        self.sigma_Gaussian = sigma_Gaussian
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

        self.regress = regress

        self.num_candidates_perframe = 147

        logging.info("preloading PC...")
        self.list_of_PCs = [None] * self.num_of_frames
        self.list_of_BBs = [None] * self.num_of_frames
        self.list_of_model_indexes = [None] * self.num_of_frames
        for index in tqdm(range(self.num_of_frames)):
            PC, box = our_dataset[index]
            new_PC = CropPC(PC, box, offset=10)
            self.list_of_PCs[index] = new_PC
            self.list_of_BBs[index] = box
            self.list_of_model_indexes[index] = self.model_database.GetIdx(box.center, box.rotation_matrix)

        logging.info("PC preloaded!")

    def __getitem__(self, index):
        anno_idx = self.GetAnnotationIndex(index)
        sample_idx = self.GetSearchSpaceIndex(index)

        if sample_idx == 0:
            sample_offsets = np.zeros(3)
        else:
            gaussian = KalmanFiltering(bnd=[1, 1, 5])
            sample_offsets = gaussian.Sample(1)[0]

        this_PC = self.list_of_PCs[anno_idx]
        this_BB = self.list_of_BBs[anno_idx]
        sample_BB = GetOffsetBB(this_BB, sample_offsets)
        sample_PC = CropAndCenterPC(
            this_PC, sample_BB, offset=self.offset_BB, scale=self.scale_BB)
        if sample_PC.NumOfPoints() <= 20:
            return self.__getitem__(np.random.randint(0, self.__len__()))
        sample_PC = RegularizePC(sample_PC, self.model)[0]

        model_idx = self.list_of_model_indexes[anno_idx]
        model_PC = PointCloud(self.model_database.pts_list[model_idx])
        model_PC = RegularizePC(model_PC, self.model)[0]

        if "GAUSSIAN" in self.regress.upper():
            score = GetScoreGaussian(sample_offsets, self.sigma_Gaussian)
        else:
            print("regress should be Gaussian. Error occured!")

        return sample_PC, model_PC, score

    def __len__(self):
        return self.num_of_frames * self.num_candidates_perframe

    def GetAnnotationIndex(self, index):
        return int(index / (self.num_candidates_perframe))

    def GetSearchSpaceIndex(self, index):
        return int(index % self.num_candidates_perframe)




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# TODO: Combine validation and training?
def Train(dataloader,
          model,
          criterion_tracking,
          criterion_completion,
          optimizer,
          epoch,
          lambda_completion=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_tracking = AverageMeter()
    loss_completion = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120)  as t:
        for i, (this_PC, model_PC, target) in t:
            # measure data loading time
            data_time.update(time.time() - end)

            this_PC = this_PC.cuda()
            model_PC = model_PC.cuda()
            target = target.float().cuda(non_blocking=True).view(-1)

            # compute output
            output, model_PC_AE = model(this_PC, model_PC)

            if lambda_completion < 1:
                loss1 = criterion_tracking(output, target)
            else:
                loss1 = torch.tensor([0]).float().cuda()

            if lambda_completion != 0:
                loss2 = criterion_completion(model_PC_AE, model_PC)
            else:
                loss2 = torch.tensor([0]).float().cuda()
            loss = loss1 + lambda_completion * loss2

            # measure accuracy and record loss
            loss_tracking.update(loss1.item(), this_PC.size(0))
            loss_completion.update(loss2.item(), this_PC.size(0))
            losses.update(loss.item(), this_PC.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            t.set_description(f'Train {epoch}: '
                              f'Time {batch_time.avg:.3f}s '
                              f'(it:{batch_time.val:.3f}s) '
                              f'Data:{data_time.avg:.3f}s '
                              f'(it:{data_time.val:.3f}s) '
                              f'Loss {losses.avg:.4f} '
                              f'(tr:{loss_tracking.avg:.4f}, '
                              f'comp:{loss_completion.avg:.0f})')
    return losses.avg


if __name__ == '__main__':

    LiDARDataPath = '/home/hao/dataset/RS128_vehicle_tracking/2020-08-18/2020-08-18-18-30-25'
    vehicle_model_path = '/home/hao/work/2023-CodeStack-DATMO/matlab/vehicle_template_2024.bin'

    model_name = "Ours"

    bneck_size = 128
    batch_size = 1


    model_fusion = 'pointcloud'     # early or late fusion (pointcloud/latent/space)
    shape_aggregation = ''      # Aggregation of shapes (first/previous/firstandprevious/all/AVG/MEDIAN/MAX)
    search_space = 'Exhaustive'     # Search space (Exhaustive/Kalman/Particle/GMM<N>)
    number_candidate = 125      # Number of candidate for Kalman, Particle or GMM search space
    reference_BB = 'current_gt'     # previous_result/previous_gt/current_gt

    lambda_completion = 1e-6

    max_num_worker = 6  # number of worker

    loglevel = 'INFO'
    models_path = './models_train/'   # Folder for models to be stored or read from.
    os.makedirs(os.path.join(models_path, model_name), exist_ok=True)
    loggerpath = os.path.join(models_path, model_name, "logging.log")
    numeric_level = getattr(logging, loglevel.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(models_path, model_name,
                             datetime.now().strftime('%Y-%m-%d %H-%M-%S.log'))),
            logging.StreamHandler()
        ])

    NN_model = Model(bneck_size, AE_chkpt_file='/home/hao/new-code/CodeStack/models_train/Pretrain_ShapeNet/model.pth.tar', chkpt_file=None).cuda()


    train_dataset = SiamseTrainDataset(NN_model, LiDARDataPath, vehicle_model_path, scale_BB = 1.25)


    dataloader_Training = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=max_num_worker,
        pin_memory=True)



    # define criterions
    criterion_tracking = torch.nn.MSELoss()
    criterion_completion = ChamferLoss()


    # define optimizer
    optimizer = torch.optim.Adam(
        NN_model.parameters(),
        lr=1e-04,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False)

    # define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, patience=3)


    # trainer(
    #     dataloader_Training,
    #     dataloader_Validation,
    #     NN_model,
    #     optimizer,
    #     scheduler,
    #     criterion_tracking,
    #     criterion_completion,
    #     model_name=model_name,
    #     lambda_completion=lambda_completion)



    logging.info("start training")

    max_epochs = 100
    best_loss = 9e99

    for epoch in range(max_epochs):
        best_model_path = os.path.join("..", "models", model_name,
                                       "model.pth.tar")

        # train for one epoch
        loss_training = Train(
            dataloader_Training,
            NN_model,
            criterion_tracking,
            criterion_completion,
            optimizer,
            epoch + 1,
            lambda_completion=lambda_completion)

        state = {
            'epoch': epoch + 1,
            'state_dict': NN_model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("..", "models", model_name), exist_ok=True)
        torch.save(
            state,
            os.path.join("..", "models", model_name,
                         "model_epoch" + str(epoch + 1) + ".pth.tar"))

        # # remember best prec@1 and save checkpoint
        # is_better = loss_validation < best_loss
        # best_loss = min(loss_validation, best_loss)
        #
        # # test the model
        # if is_better:
        #     torch.save(state, best_model_path)
        #
        #
        # # update the LR scheduler
        # if scheduler is not None:
        #     prevLR = optimizer.param_groups[0]['lr']
        #     scheduler.step(loss_validation)
        #     currLR = optimizer.param_groups[0]['lr']
        #     if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
        #         logging.info("Plateau Reached!")
        #
        #     if (prevLR < 2 * scheduler.eps and
        #             scheduler.num_bad_epochs >= scheduler.patience):
        #         logging.info(
        #             "Plateau Reached and no more reduction -> Exiting Loop")
        #         break
