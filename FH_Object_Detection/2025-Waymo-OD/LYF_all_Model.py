from collections import OrderedDict
from functools import reduce
import copy

import numpy as np
import spconv
import spconv.pytorch
import torch
import torch.utils.checkpoint as cp
import torch_scatter
from spconv.core import ConvAlgo
from spconv.pytorch import SparseConv2d
from spconv.pytorch import SparseSequential
from torch import nn
from torch.nn import functional as F

from centerloss import FastFocalLoss
from centerloss import IouLoss
from centerloss import IouRegLoss
from centerloss import RegLoss
from det3d.iou3d_nms import iou3d_nms_cuda


def ReplaceFeature(out, new_features):
    if "replace_feature" in out.__dir__():
        return out.replace_feature(new_features)

    out.features = new_features
    return out


class PFNLayer(nn.Module):  # reader 初始化时构建 PFN 层序列使用

    def __init__(self, in_channels, out_channels, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs, unq_inv):
        torch.backends.cudnn.enabled = False
        x = self.linear(inputs)
        x = self.norm(x)
        x = F.relu(x)
        torch.backends.cudnn.enabled = True

        feat_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        x_max = feat_max[unq_inv]

        if self.last_vfe:
            return x_max

        return torch.cat([x, x_max], dim=1)


class PillarNet(nn.Module):  # reader 初始化时构建 pillar voxelization 使用

    def __init__(self, num_input_features, voxel_size, pc_range):
        super().__init__()
        self.num_input_features = num_input_features
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)

    def forward(self, points):
        device = points.device
        dtype = points.dtype

        grid_size = (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int64)

        voxel_size = torch.from_numpy(self.voxel_size).type_as(points).to(device)
        pc_range = torch.from_numpy(self.pc_range).type_as(points).to(device)

        points_coords = (points[:, 1:4] - pc_range[:3].view(-1, 3)) / voxel_size.view(-1, 3)
        mask = reduce(
            torch.logical_and,
            (
                points_coords[:, 0] >= 0,
                points_coords[:, 0] < grid_size[0],
                points_coords[:, 1] >= 0,
                points_coords[:, 1] < grid_size[1],
            ),
        )

        points = points[mask]
        points_coords = points_coords[mask]

        points_coords = points_coords.long()
        batch_idx = points[:, 0:1].long()

        points_index = torch.cat((batch_idx, points_coords[:, :2]), dim=1)
        unq, unq_inv = torch.unique(points_index, return_inverse=True, dim=0)
        unq = unq.int()

        points_mean_scatter = torch_scatter.scatter_mean(points[:, 1:4], unq_inv, dim=0)
        f_cluster = points[:, 1:4] - points_mean_scatter[unq_inv]
        f_center = points[:, 1:3] - (
            points_coords[:, :2].to(dtype) * voxel_size[:2].unsqueeze(0)
            + voxel_size[:2].unsqueeze(0) / 2
            + pc_range[:2].unsqueeze(0)
        )

        features = torch.cat([points[:, 1:], f_cluster, f_center], dim=-1)

        return features, unq[:, [0, 2, 1]], unq_inv, grid_size[[1, 0]]


class SparseConvBlock(spconv.pytorch.SparseModule):  # backbone stage 初始化时使用的基础稀疏卷积块

    def __init__(self, in_channels, out_channels, kernel_size, stride, use_subm=True, bias=False):
        super().__init__()
        if stride == 1 and use_subm:
            self.conv = spconv.pytorch.SubMConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=bias,
            )
        else:
            self.conv = spconv.pytorch.SparseConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                bias=bias,
            )

        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = ReplaceFeature(out, self.norm(out.features))
        out = ReplaceFeature(out, self.act(out.features))
        return out


class SparseBasicBlock(spconv.pytorch.SparseModule):  # backbone stage 初始化时使用的残差稀疏卷积块

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.block1 = SparseConvBlock(channels, channels, kernel_size, 1)
        self.conv2 = spconv.pytorch.SubMConv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            stride=1,
            bias=False,
            algo=ConvAlgo.Native,
        )
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.act2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.conv2(out)
        out = ReplaceFeature(out, self.norm2(out.features))
        out = ReplaceFeature(out, out.features + identity.features)
        out = ReplaceFeature(out, self.act2(out.features))
        return out


class ConvBlock(nn.Module):  # neck 和 head 初始化时都会复用的二维卷积块

    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        stride=1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        padding = kwargs.get('padding', kernel_size // 2)
        self.conv = conv_layer(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = norm_layer(planes)
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class BasicBlock(nn.Module):  # neck 初始化时作为 ASPP 前置卷积块使用

    def __init__(self, inplanes, kernel_size=3):
        super().__init__()
        self.block1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.block2 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.act(out)
        return out


class SepHead(nn.Module):  # head 初始化时为每个检测 task 构建独立预测头使用

    def __init__(
        self,
        in_channels,
        heads,
        stride=1,
        head_conv=64,
        final_kernel=1,
        bn=True,
        init_bias=-2.19,
    ):
        super().__init__()
        if stride > 1:
            self.deblock = ConvBlock(
                in_channels,
                head_conv,
                kernel_size=int(stride),
                stride=int(stride),
                padding=0,
                conv_layer=nn.ConvTranspose2d,
            )
            in_channels = head_conv
        else:
            self.deblock = nn.Identity()

        self.heads = heads
        for head_name in self.heads:
            classes, num_conv = self.heads[head_name]
            fc = nn.Sequential()
            for _ in range(num_conv - 1):
                fc.append(
                    nn.Conv2d(
                        in_channels,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True,
                    )
                )
                if bn:
                    fc.append(nn.BatchNorm2d(head_conv))
                fc.append(nn.ReLU())

            fc.append(
                nn.Conv2d(
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )

            if 'hm' in head_name:
                fc[-1].bias.data.fill_(init_bias)

            self.__setattr__(head_name, fc)

    def forward(self, x):
        x = self.deblock(x)
        ret_dict = {}
        for head_name in self.heads:
            ret_dict[head_name] = self.__getattr__(head_name)(x)
        return ret_dict


def RotateNmsPcdet(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))

    if len(boxes) == 0:
        num_out = 0
    else:
        num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)

    selected = order[keep[:num_out].cuda()].contiguous()
    if post_max_size is not None:
        selected = selected[:post_max_size]
    return selected


class LYF_all_Model(nn.Module):

    def __init__(self, cfg, common_variables):
        super().__init__()
        self.post_processing = copy.deepcopy(cfg.post_processing)
        self.post_processing.update(common_variables)

        self.tasks = common_variables['tasks']

        ## reader 初始化
        num_input_features = cfg.reader.num_input_features
        num_filters = list(cfg.reader.num_filters)
        voxel_size = cfg.reader.get('voxel_size', common_variables['voxel_size'])
        pc_range = cfg.reader.get('pc_range', common_variables['pc_range'])

        self.reader_voxelization = PillarNet(num_input_features, voxel_size, pc_range)

        pfn_input_channels = num_input_features + 5
        pfn_channels = [pfn_input_channels] + num_filters
        pfn_layers = []
        for layer_idx in range(len(pfn_channels) - 1):
            pfn_layers.append(
                PFNLayer(
                    pfn_channels[layer_idx],
                    pfn_channels[layer_idx + 1],
                    last_layer=(layer_idx == len(pfn_channels) - 2),
                )
            )
        self.reader_pfn_layers = nn.ModuleList(pfn_layers)
        self.reader_output_channels = num_filters[-1]

        ## backbone 初始化
        layer_nums = list(cfg.backbone.layer_nums)
        ds_layer_strides = list(cfg.backbone.ds_layer_strides)
        ds_num_filters = list(cfg.backbone.ds_num_filters)
        kernel_size = list(cfg.backbone.get('kernel_size', [3, 3, 3, 3]))

        assert len(layer_nums) == len(ds_layer_strides)
        assert len(layer_nums) == len(ds_num_filters)

        in_channels = cfg.backbone.get('num_input_features', self.reader_output_channels)
        stage_input_channels = [in_channels, *ds_num_filters[:-1]]

        backbone_blocks = []
        for stage_idx, num_blocks in enumerate(layer_nums):
            backbone_blocks.append(
                self._MakeSparseStage(
                    stage_input_channels[stage_idx],
                    ds_num_filters[stage_idx],
                    kernel_size[stage_idx],
                    ds_layer_strides[stage_idx],
                    num_blocks,
                )
            )
        self.backbone_stages = nn.ModuleList(backbone_blocks)

        self.backbone_mapping = SparseSequential(
            SparseConv2d(ds_num_filters[-1], ds_num_filters[-1], 1, 1, bias=False),
            nn.BatchNorm1d(ds_num_filters[-1], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.backbone_output_channels = ds_num_filters[-1]

        ## neck 初始化
        if cfg.neck is None:
            self.b_use_neck = False
            self.neck_pre_conv = None
            self.neck_weight = None
            self.neck_post_conv = None
        else:
            self.b_use_neck = True
            in_channels = cfg.neck.in_channels
            self.neck_pre_conv = BasicBlock(in_channels)
            self.neck_weight = nn.Parameter(torch.randn(in_channels, in_channels, 3, 3))
            self.neck_post_conv = ConvBlock(in_channels * 4, in_channels, kernel_size=1, stride=1)

        ## head 初始化
        self.head_code_weights = cfg.head.code_weights
        self.head_weight = cfg.head.weight
        self.head_num_classes = [len(task) for task in self.tasks]
        self.head_strides = list(cfg.head.strides)
        self.head_rectifier = cfg.head.get('rectifier', [[0.], [0.], [0.]])
        self.head_common_heads = copy.deepcopy(cfg.head.common_heads)
        self.head_voxel_size = cfg.head.get('voxel_size', common_variables['voxel_size'])
        self.head_pc_range = cfg.head.get('pc_range', common_variables['pc_range'])
        self.head_out_size_factor = cfg.head.get('out_size_factor', common_variables['out_size_factor'])
        self.b_with_reg_iou = cfg.head.get('with_reg_iou', False)
        self.b_with_iou = 'iou' in self.head_common_heads

        share_conv_channel = cfg.head.get('share_conv_channel', 64)
        init_bias = cfg.head.get('init_bias', -2.19)
        num_hm_conv = cfg.head.get('num_hm_conv', 2)
        in_channels = cfg.head.in_channels

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()
        if self.b_with_iou:
            self.crit_iou = IouLoss()
        if self.b_with_reg_iou:
            self.crit_iou_reg = IouRegLoss()

        self.head_shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True),
        )

        self.head_tasks = nn.ModuleList()
        for num_cls, stride in zip(self.head_num_classes, self.head_strides):
            task_heads = copy.deepcopy(self.head_common_heads)
            task_heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.head_tasks.append(
                SepHead(
                    share_conv_channel,
                    task_heads,
                    stride=stride,
                    bn=True,
                    init_bias=init_bias,
                    final_kernel=3,
                )
            )

    def _MakeSparseStage(self, inplanes, planes, kernel_size, stride, num_blocks):
        layers = [
            SparseConvBlock(
                inplanes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                use_subm=False,
            )
        ]
        for _ in range(num_blocks):
            layers.append(SparseBasicBlock(planes, kernel_size=kernel_size))
        return spconv.pytorch.SparseSequential(*layers)

    def _sigmoid(self, x):
        return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)

    def ForwardNetwork(self, example):
        points = example['points']

        ## reader forward
        features, coords, unq_inv, grid_size = self.reader_voxelization(points)
        for pfn_layer in self.reader_pfn_layers:
            features = pfn_layer(features, unq_inv)
        pillar_features = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]

        ## backbone forward
        batch_size = len(torch.unique(coords[:, 0]))
        x = spconv.pytorch.SparseConvTensor(pillar_features, coords, grid_size, batch_size)
        for stage in self.backbone_stages:
            x = stage(x)
        x = self.backbone_mapping(x)
        x = x.dense()

        ## neck forward
        if self.b_use_neck:
            def NeckForward(feat):
                feat = self.neck_pre_conv(feat)
                branch1 = F.conv2d(feat, self.neck_weight, stride=1, bias=None, padding=1, dilation=1)
                branch6 = F.conv2d(feat, self.neck_weight, stride=1, bias=None, padding=6, dilation=6)
                branch12 = F.conv2d(feat, self.neck_weight, stride=1, bias=None, padding=12, dilation=12)
                branch18 = F.conv2d(feat, self.neck_weight, stride=1, bias=None, padding=18, dilation=18)
                return self.neck_post_conv(torch.cat((branch1, branch6, branch12, branch18), dim=1))

            if x.requires_grad:
                x = cp.checkpoint(NeckForward, x)
            else:
                x = NeckForward(x)

        ## head forward
        preds = []
        x = self.head_shared_conv(x)
        for task in self.head_tasks:
            preds.append(task(x))

        return preds

    def ComputeLoss(self, example, preds_dicts):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])
            hm_loss = self.crit(
                preds_dict['hm'],
                example['hm'][task_id],
                example['ind'][task_id],
                example['mask'][task_id],
                example['cat'][task_id],
            )

            target_box = example['anno_box'][task_id]
            preds_dict['anno_box'] = torch.cat(
                (
                    preds_dict['reg'],
                    preds_dict['height'],
                    preds_dict['dim'],
                    preds_dict['rot'],
                ),
                dim=1,
            )

            box_loss = self.crit_reg(
                preds_dict['anno_box'],
                example['mask'][task_id],
                example['ind'][task_id],
                target_box,
            )
            loc_loss = (box_loss * box_loss.new_tensor(self.head_code_weights)).sum()
            loss = hm_loss + self.head_weight * loc_loss

            ret = OrderedDict()
            ret.update(
                {
                    'task': self.tasks[task_id],
                    'loss': loss,
                    'hm_loss': hm_loss.detach().cpu(),
                    'loc_loss': loc_loss.detach().cpu(),
                    'loc_loss_elem': box_loss.detach().cpu(),
                    'num_positive': example['mask'][task_id].float().sum().cpu(),
                }
            )

            if self.b_with_iou or self.b_with_reg_iou:
                batch_dim = torch.exp(torch.clamp(preds_dict['dim'], min=-5, max=5))
                batch_dim = batch_dim.permute(0, 2, 3, 1).contiguous()
                batch_rot = preds_dict['rot'].clone().permute(0, 2, 3, 1).contiguous()
                batch_rots = batch_rot[..., 0:1]
                batch_rotc = batch_rot[..., 1:2]
                batch_rot = torch.atan2(batch_rots, batch_rotc)
                batch_reg = preds_dict['reg'].clone().permute(0, 2, 3, 1).contiguous()
                batch_hei = preds_dict['height'].clone().permute(0, 2, 3, 1).contiguous()

                batch, height, width, _ = batch_dim.size()
                batch_reg = batch_reg.reshape(batch, height * width, 2)
                batch_hei = batch_hei.reshape(batch, height * width, 1)
                batch_rot = batch_rot.reshape(batch, height * width, 1)
                batch_dim = batch_dim.reshape(batch, height * width, 3)

                ys, xs = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
                ys = ys.view(1, height, width).repeat(batch, 1, 1).to(batch_dim)
                xs = xs.view(1, height, width).repeat(batch, 1, 1).to(batch_dim)

                xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
                ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

                xs = xs * self.head_out_size_factor[task_id] * self.head_voxel_size[0] + self.head_pc_range[0]
                ys = ys * self.head_out_size_factor[task_id] * self.head_voxel_size[1] + self.head_pc_range[1]

                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
                batch_box_preds = batch_box_preds.permute(0, 2, 1).contiguous().reshape(batch, -1, height, width)

                if self.b_with_iou:
                    pred_boxes_for_iou = batch_box_preds.detach()
                    iou_loss = self.crit_iou(
                        preds_dict['iou'],
                        example['mask'][task_id],
                        example['ind'][task_id],
                        pred_boxes_for_iou,
                        example['gt_boxes'][task_id],
                    )
                    loss = loss + iou_loss
                    ret.update({'iou_loss': iou_loss.detach().cpu()})

                if self.b_with_reg_iou:
                    iou_reg_loss = self.crit_iou_reg(
                        batch_box_preds,
                        example['mask'][task_id],
                        example['ind'][task_id],
                        example['gt_boxes'][task_id],
                    )
                    loss = loss + self.head_weight * iou_reg_loss
                    ret.update({'iou_reg_loss': iou_reg_loss.detach().cpu()})

            rets.append(ret)
            if task_id == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        return total_loss, rets

    def TrainingStep(self, example):
        preds = self.ForwardNetwork(example)
        loss, log_vars = self.ComputeLoss(example, preds)
        return loss, log_vars

    @torch.no_grad()
    def ValidationStep(self, example):
        preds = self.ForwardNetwork(example)
        outputs = self.Predict(example, preds, self.post_processing)

        detections = {}
        for output in outputs:
            token = output["token"]
            for key, value in output.items():
                if key != "token":
                    output[key] = value.to(torch.device("cpu"))
            detections[token] = output

        return detections

    @torch.no_grad()
    def Predict(self, example, preds_dicts, test_cfg):
        rets = []
        metas = []

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]
            if "token" not in example or len(example["token"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["token"]

            batch_hm = torch.sigmoid(preds_dict['hm'])
            batch_dim = torch.exp(preds_dict['dim'])
            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if 'iou' in preds_dict:
                batch_iou = (preds_dict['iou'].squeeze(dim=-1) + 1) * 0.5
                batch_iou = batch_iou.type_as(batch_dim)
            else:
                batch_iou = torch.ones(
                    (batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                    dtype=batch_dim.dtype,
                ).to(batch_hm.device)

            batch_rot = torch.atan2(batch_rots, batch_rotc)
            batch, height, width, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, height * width, 2)
            batch_hei = batch_hei.reshape(batch, height * width, 1)
            batch_rot = batch_rot.reshape(batch, height * width, 1)
            batch_dim = batch_dim.reshape(batch, height * width, 3)
            batch_hm = batch_hm.reshape(batch, height * width, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
            ys = ys.view(1, height, width).repeat(batch, 1, 1).to(batch_hm.device).float()
            xs = xs.view(1, height, width).repeat(batch, 1, 1).to(batch_hm.device).float()

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * test_cfg.out_size_factor[task_id] * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor[task_id] * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)
            rets.append(
                self.PostProcessing(
                    task_id,
                    batch_box_preds,
                    batch_hm,
                    test_cfg,
                    post_center_range,
                    batch_iou,
                )
            )

        ret_list = []
        num_samples = len(rets[0])
        for sample_idx in range(num_samples):
            ret = {}
            for key in rets[0][sample_idx].keys():
                if key in ["box3d_lidar", "scores"]:
                    ret[key] = torch.cat([ret_per_task[sample_idx][key] for ret_per_task in rets])
                elif key in ["label_preds"]:
                    flag = 0
                    for task_idx, num_class in enumerate(self.head_num_classes):
                        rets[task_idx][sample_idx][key] += flag
                        flag += num_class
                    ret[key] = torch.cat([ret_per_task[sample_idx][key] for ret_per_task in rets])

            ret['token'] = metas[0][sample_idx]
            ret_list.append(ret)

        return ret_list

    @torch.no_grad()
    def PostProcessing(self, task_id, batch_box_preds, batch_hm, test_cfg, post_center_range, batch_iou):
        batch_size = len(batch_hm)
        prediction_dicts = []

        for batch_idx in range(batch_size):
            box_preds = batch_box_preds[batch_idx]
            hm_preds = batch_hm[batch_idx]
            iou_preds = batch_iou[batch_idx].view(-1)

            scores, labels = torch.max(hm_preds, dim=-1)
            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) & (
                box_preds[..., :3] <= post_center_range[3:]
            ).all(1)
            mask = distance_mask & score_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            iou_preds = torch.clamp(iou_preds[mask], min=0.0, max=1.0)

            rectifier = torch.tensor(self.head_rectifier[task_id]).to(hm_preds)
            scores = torch.pow(scores, 1 - rectifier[labels]) * torch.pow(iou_preds, rectifier[labels])

            selected_boxes = torch.zeros((0, 7)).to(box_preds)
            selected_labels = torch.zeros((0,), dtype=torch.int64).to(labels)
            selected_scores = torch.zeros((0,)).to(scores)

            for class_id in range(hm_preds.shape[-1]):
                scores_class = scores[labels == class_id]
                labels_class = labels[labels == class_id]
                box_preds_class = box_preds[labels == class_id]
                selected = RotateNmsPcdet(
                    box_preds_class,
                    scores_class,
                    thresh=test_cfg.nms.nms_iou_threshold[task_id][class_id],
                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                    post_max_size=test_cfg.nms.nms_post_max_size,
                )

                selected_boxes = torch.cat((selected_boxes, box_preds_class[selected]), dim=0)
                selected_scores = torch.cat((selected_scores, scores_class[selected]), dim=0)
                selected_labels = torch.cat((selected_labels, labels_class[selected]), dim=0)

            prediction_dicts.append(
                {
                    'box3d_lidar': selected_boxes,
                    'scores': selected_scores,
                    'label_preds': selected_labels,
                }
            )

        return prediction_dicts


    def forward(self, example):
        if self.training:
            return self.TrainingStep(example)
        return self.ValidationStep(example)
