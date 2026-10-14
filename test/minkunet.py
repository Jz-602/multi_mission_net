import torch
import torch.nn as nn
import spconv.pytorch as spconv


__all__ = ['MinkUNet']


class BaseSegmentor(nn.Module):
    def __init__(self, model_cfgs, num_class):
        super().__init__()
        self.model_cfgs = model_cfgs
        self.num_class = num_class


def ReplaceFeature(x, new_features):
    if hasattr(x, 'replace_feature'):
        return x.replace_feature(new_features)
    x.features = new_features
    return x


def BuildCoordHash(coords, max_vals):
    coords = coords.long()
    if coords.numel() == 0:
        return torch.zeros((0,), device=coords.device, dtype=torch.long)

    hash_values = coords[:, 0]
    for dim_idx in range(1, coords.shape[1]):
        hash_values = hash_values * max_vals[dim_idx] + coords[:, dim_idx]
    return hash_values


def AlignFeaturesByCoords(reference_coords, source_coords, source_features):
    if reference_coords.numel() == 0:
        return torch.zeros(
            (0, source_features.shape[1]),
            device=source_features.device,
            dtype=source_features.dtype,
        )

    max_vals = torch.maximum(
        reference_coords.long().max(dim=0).values,
        source_coords.long().max(dim=0).values,
    ) + 1
    source_hash = BuildCoordHash(source_coords, max_vals)
    reference_hash = BuildCoordHash(reference_coords, max_vals)

    sorted_hash, sorted_order = torch.sort(source_hash)
    search_pos = torch.searchsorted(sorted_hash, reference_hash)
    aligned = torch.zeros(
        (reference_coords.shape[0], source_features.shape[1]),
        device=source_features.device,
        dtype=source_features.dtype,
    )

    valid_mask = search_pos < sorted_hash.shape[0]
    if valid_mask.any():
        valid_pos = search_pos[valid_mask]
        matched_mask = sorted_hash[valid_pos] == reference_hash[valid_mask]
        if matched_mask.any():
            matched_pos = valid_pos[matched_mask]
            matched_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)[matched_mask]
            aligned[matched_indices] = source_features[sorted_order[matched_pos]]
    return aligned


def CatSparseTensors(x, skip_x):
    skip_features = AlignFeaturesByCoords(x.indices, skip_x.indices, skip_x.features)
    return ReplaceFeature(x, torch.cat([x.features, skip_features], dim=1))


def GatherPointWiseFeatures(reference_indices, sparse_tensor, stride_factor=1):
    query_indices = reference_indices.clone()
    if stride_factor > 1:
        query_indices[:, 1:] = torch.div(
            query_indices[:, 1:],
            stride_factor,
            rounding_mode='floor',
        )
    return AlignFeaturesByCoords(query_indices, sparse_tensor.indices, sparse_tensor.features)


def BuildInputSparseTensor(batch_dict):
    voxel_dict = batch_dict['voxels']
    if isinstance(voxel_dict, spconv.SparseConvTensor):
        return voxel_dict

    if 'sparse_tensor' in voxel_dict:
        return voxel_dict['sparse_tensor']

    features = voxel_dict['voxels']
    indices = voxel_dict['coors'].int()
    spatial_shape = voxel_dict['spatial_shape']
    batch_size = voxel_dict['batch_size']
    return spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=batch_size,
    )


class SyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, input):
        return ReplaceFeature(input, super().forward(input.features))


class BatchNorm(nn.BatchNorm1d):
    def forward(self, input):
        return ReplaceFeature(input, super().forward(input.features))


def BuildSparseConv(inc, outc, ks, stride=1, dilation=1, transposed=False, indice_key=None, bias=False):
    if transposed:
        if indice_key is not None:
            return spconv.SparseInverseConv3d(
                inc,
                outc,
                kernel_size=ks,
                indice_key=indice_key,
                bias=bias,
            )
        return spconv.SparseConvTranspose3d(
            inc,
            outc,
            kernel_size=ks,
            stride=stride,
            bias=bias,
        )

    if stride == 1:
        return spconv.SubMConv3d(
            inc,
            outc,
            kernel_size=ks,
            stride=1,
            dilation=dilation,
            padding=(ks // 2) * dilation,
            indice_key=indice_key,
            bias=bias,
        )

    return spconv.SparseConv3d(
        inc,
        outc,
        kernel_size=ks,
        stride=stride,
        dilation=dilation,
        padding=0 if ks == 2 else (ks // 2) * dilation,
        indice_key=indice_key,
        bias=bias,
    )


class BasicConvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
        indice_key: str = None,
    ):
        super().__init__()
        self.conv = BuildSparseConv(
            inc,
            outc,
            ks,
            stride=stride,
            dilation=dilation,
            indice_key=indice_key,
            bias=False,
        )
        self.norm = SyncBatchNorm(outc) if if_dist else BatchNorm(outc)
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = ReplaceFeature(out, self.act(out.features))
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        if_dist: bool = False,
        indice_key: str = None,
    ):
        super().__init__()
        self.conv = BuildSparseConv(
            inc,
            outc,
            ks,
            stride=stride,
            transposed=True,
            indice_key=indice_key,
            bias=False,
        )
        self.norm = SyncBatchNorm(outc) if if_dist else BatchNorm(outc)
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = ReplaceFeature(out, self.act(out.features))
        return out


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
        indice_key: str = None,
    ):
        super().__init__()
        self.conv1 = BuildSparseConv(
            inc,
            outc,
            ks,
            stride=stride,
            dilation=dilation,
            indice_key=indice_key,
            bias=False,
        )
        self.bn1 = SyncBatchNorm(outc) if if_dist else BatchNorm(outc)
        self.conv2 = BuildSparseConv(
            outc,
            outc,
            ks,
            stride=1,
            dilation=dilation,
            indice_key=indice_key,
            bias=False,
        )
        self.bn2 = SyncBatchNorm(outc) if if_dist else BatchNorm(outc)

        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                BuildSparseConv(
                    inc,
                    outc * self.expansion,
                    ks=1,
                    stride=stride,
                    indice_key=None if stride > 1 else indice_key,
                    bias=False,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x if isinstance(self.downsample, nn.Identity) else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = ReplaceFeature(out, self.relu(out.features))

        out = self.conv2(out)
        out = self.bn2(out)
        out = ReplaceFeature(out, out.features + identity.features)
        out = ReplaceFeature(out, self.relu(out.features))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
        indice_key: str = None,
    ):
        super().__init__()
        self.conv1 = BuildSparseConv(
            inc,
            outc,
            ks=1,
            stride=1,
            indice_key=indice_key,
            bias=False,
        )
        self.bn1 = SyncBatchNorm(outc) if if_dist else BatchNorm(outc)
        self.conv2 = BuildSparseConv(
            outc,
            outc,
            ks=ks,
            stride=stride,
            dilation=dilation,
            indice_key=indice_key,
            bias=False,
        )
        self.bn2 = SyncBatchNorm(outc) if if_dist else BatchNorm(outc)
        self.conv3 = BuildSparseConv(
            outc,
            outc * self.expansion,
            ks=1,
            stride=1,
            indice_key=indice_key,
            bias=False,
        )
        self.bn3 = SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion)

        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                BuildSparseConv(
                    inc,
                    outc * self.expansion,
                    ks=1,
                    stride=stride,
                    indice_key=None if stride > 1 else indice_key,
                    bias=False,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x if isinstance(self.downsample, nn.Identity) else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = ReplaceFeature(out, self.relu(out.features))

        out = self.conv2(out)
        out = self.bn2(out)
        out = ReplaceFeature(out, self.relu(out.features))

        out = self.conv3(out)
        out = self.bn3(out)
        out = ReplaceFeature(out, out.features + identity.features)
        out = ReplaceFeature(out, self.relu(out.features))
        return out


class MinkUNet(BaseSegmentor):
    def __init__(
        self,
        model_cfgs,
        num_class: int,
    ):
        super().__init__(model_cfgs, num_class)
        self.in_feature_dim = model_cfgs.IN_FEATURE_DIM
        self.num_layer = model_cfgs.get('NUM_LAYER', [2, 3, 4, 6, 2, 2, 2, 2])
        self.block = {
            'ResBlock': ResidualBlock,
            'ResidualBlock': ResidualBlock,
            'Bottleneck': Bottleneck,
        }[model_cfgs.get('BLOCK', 'ResidualBlock')]

        cr = model_cfgs.get('cr', 1.0)
        cs = model_cfgs.get('PLANES', [32, 32, 64, 128, 256, 256, 128, 96, 96])
        cs = [int(cr * x) for x in cs]

        self.pres = model_cfgs.get('pres', 0.05)
        self.vres = model_cfgs.get('vres', 0.05)

        self.stem = nn.Sequential(
            BasicConvolutionBlock(
                self.in_feature_dim,
                cs[0],
                ks=3,
                stride=1,
                if_dist=model_cfgs.IF_DIST,
                indice_key='subm0',
            ),
            BasicConvolutionBlock(
                cs[0],
                cs[0],
                ks=3,
                stride=1,
                if_dist=model_cfgs.IF_DIST,
                indice_key='subm0',
            ),
        )

        self.in_channels = cs[0]
        if_dist = model_cfgs.IF_DIST

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels,
                self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
                indice_key='spconv1',
            ),
            *self._make_layer(
                self.block,
                cs[1],
                self.num_layer[0],
                if_dist=if_dist,
                indice_key='subm1',
            ),
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels,
                self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
                indice_key='spconv2',
            ),
            *self._make_layer(
                self.block,
                cs[2],
                self.num_layer[1],
                if_dist=if_dist,
                indice_key='subm2',
            ),
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels,
                self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
                indice_key='spconv3',
            ),
            *self._make_layer(
                self.block,
                cs[3],
                self.num_layer[2],
                if_dist=if_dist,
                indice_key='subm3',
            ),
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels,
                self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
                indice_key='spconv4',
            ),
            *self._make_layer(
                self.block,
                cs[4],
                self.num_layer[3],
                if_dist=if_dist,
                indice_key='subm4',
            ),
        )

        self.up1 = [
            BasicDeconvolutionBlock(
                self.in_channels,
                cs[5],
                ks=2,
                stride=2,
                if_dist=if_dist,
                indice_key='spconv4',
            )
        ]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(
            nn.Sequential(
                *self._make_layer(
                    self.block,
                    cs[5],
                    self.num_layer[4],
                    if_dist=if_dist,
                    indice_key='subm3_up',
                )
            )
        )
        self.up1 = nn.ModuleList(self.up1)

        self.up2 = [
            BasicDeconvolutionBlock(
                self.in_channels,
                cs[6],
                ks=2,
                stride=2,
                if_dist=if_dist,
                indice_key='spconv3',
            )
        ]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(
            nn.Sequential(
                *self._make_layer(
                    self.block,
                    cs[6],
                    self.num_layer[5],
                    if_dist=if_dist,
                    indice_key='subm2_up',
                )
            )
        )
        self.up2 = nn.ModuleList(self.up2)

        self.up3 = [
            BasicDeconvolutionBlock(
                self.in_channels,
                cs[7],
                ks=2,
                stride=2,
                if_dist=if_dist,
                indice_key='spconv2',
            )
        ]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(
            nn.Sequential(
                *self._make_layer(
                    self.block,
                    cs[7],
                    self.num_layer[6],
                    if_dist=if_dist,
                    indice_key='subm1_up',
                )
            )
        )
        self.up3 = nn.ModuleList(self.up3)

        self.up4 = [
            BasicDeconvolutionBlock(
                self.in_channels,
                cs[8],
                ks=2,
                stride=2,
                if_dist=if_dist,
                indice_key='spconv1',
            )
        ]
        self.in_channels = cs[8] + cs[0]
        self.up4.append(
            nn.Sequential(
                *self._make_layer(
                    self.block,
                    cs[8],
                    self.num_layer[7],
                    if_dist=if_dist,
                    indice_key='subm0_up',
                )
            )
        )
        self.up4 = nn.ModuleList(self.up4)

        self.classifier = nn.Linear(
            (cs[4] + cs[6] + cs[8]) * self.block.expansion,
            self.num_class,
        )

        self.weight_initialization()

        dropout_p = model_cfgs.get('DROPOUT_P', 0.3)
        self.dropout = nn.Dropout(dropout_p, True)
        self.ignore_label = model_cfgs.get('IGNORE_LABEL', -1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)

    def _make_layer(self, block, out_channels, num_block, stride=1, if_dist=False, indice_key=None):
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride=stride, if_dist=if_dist, indice_key=indice_key)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(
                block(self.in_channels, out_channels, if_dist=if_dist, indice_key=indice_key)
            )
        return layers

    def weight_initialization(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, batch_dict, return_logit=False, return_tta=False):
        x0 = BuildInputSparseTensor(batch_dict)
        x0 = ReplaceFeature(x0, x0.features[:, :self.in_feature_dim])
        reference_indices = x0.indices.clone()
        x0 = self.stem(x0)

        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = GatherPointWiseFeatures(reference_indices, x4, stride_factor=16)

        x4 = ReplaceFeature(x4, self.dropout(x4.features))
        y1 = self.up1[0](x4)
        y1 = CatSparseTensors(y1, x3)
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = CatSparseTensors(y2, x2)
        y2 = self.up2[1](y2)
        z2 = GatherPointWiseFeatures(reference_indices, y2, stride_factor=4)

        y2 = ReplaceFeature(y2, self.dropout(y2.features))
        y3 = self.up3[0](y2)
        y3 = CatSparseTensors(y3, x1)
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = CatSparseTensors(y4, x0)
        y4 = self.up4[1](y4)
        z3 = GatherPointWiseFeatures(reference_indices, y4, stride_factor=1)

        out = self.classifier(torch.cat([z1, z2, z3], dim=1))

        if self.training:
            target_tensor = batch_dict['targets']
            target = target_tensor.F.long() if hasattr(target_tensor, 'F') else target_tensor.long()
            target = target.to(out.device, non_blocking=True)
            loss = self.criterion(out, target)

            ret_dict = {'loss': loss}
            tb_dict = {'loss': float(loss.detach().cpu())}
            disp_dict = {'loss': float(loss.detach().cpu())}
            return ret_dict, tb_dict, disp_dict

        if 'inverse_map' not in batch_dict or 'targets_mapped' not in batch_dict:
            pred = out.softmax(1) if (return_logit or return_tta) else out.argmax(1)
            return {'point_predict': pred, 'point_predict_logits': out}

        invs = batch_dict['inverse_map']
        all_labels = batch_dict['targets_mapped']
        point_predict = []
        point_labels = []
        point_predict_logits = []
        point_idx_in_map = batch_dict.get('idx', None)

        batch_ids = reference_indices[:, 0].long()
        max_batch = int(invs.indices[:, 0].max().item()) + 1 if hasattr(invs, 'indices') else int(invs.C[:, -1].max().item()) + 1
        for idx in range(max_batch):
            cur_scene_pts = batch_ids == idx
            if hasattr(invs, 'features'):
                cur_inv = invs.features[invs.indices[:, 0] == idx].long().view(-1)
                cur_label = all_labels.indices[:, 0] == idx
                targets_mapped = all_labels.features[cur_label].view(-1)
            else:
                cur_inv = invs.F[invs.C[:, -1] == idx].long()
                cur_label = all_labels.C[:, -1] == idx
                targets_mapped = all_labels.F[cur_label]

            outputs_logits = out[cur_scene_pts][cur_inv]
            outputs_mapped = outputs_logits.softmax(1) if (return_logit or return_tta) else outputs_logits.argmax(1)
            num_points = batch_dict['num_points'][idx] if 'num_points' in batch_dict else outputs_mapped.shape[0]

            point_predict.append(outputs_mapped[:num_points].detach().cpu().numpy())
            point_labels.append(targets_mapped[:num_points].detach().cpu().numpy())
            point_predict_logits.append(outputs_logits[:num_points].detach().cpu().numpy())

        return {
            'point_predict': point_predict,
            'point_labels': point_labels,
            'name': batch_dict.get('name', None),
            'point_predict_logits': point_predict_logits,
            'point_idx_in_map': point_idx_in_map,
        }


if __name__ == '__main__':
    class Config(dict):
        def __getattr__(self, key):
            return self[key]

        def get(self, key, default=None):
            return super().get(key, default)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cfgs = Config(
        IN_FEATURE_DIM=4,
        IF_DIST=False,
        BLOCK='ResidualBlock',
        IGNORE_LABEL=-1,
    )
    model = MinkUNet(model_cfgs, num_class=6).to(device)

    def BuildSparseTensor(device):
        features = torch.tensor(
            [
                [1.0, 0.1, 0.2, 0.3],
                [2.0, 0.2, 0.3, 0.4],
                [3.0, 0.3, 0.4, 0.5],
                [4.0, 0.4, 0.5, 0.6],
                [5.0, 0.5, 0.6, 0.7],
                [6.0, 0.6, 0.7, 0.8],
                [7.0, 0.7, 0.8, 0.9],
                [8.0, 0.8, 0.9, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        indices = torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 0, 1, 3],
                [0, 1, 2, 2],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 2, 3, 2],
                [1, 2, 0, 1],
                [1, 2, 1, 1],
            ],
            dtype=torch.int32,
            device=device,
        )

        spatial_shape = [6, 6, 6]
        batch_size = 2
        return spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )

    lidar = BuildSparseTensor(device)
    labels = torch.randint(0, 6, (lidar.features.shape[0],), device=device)

    batch_dict = {
        'voxels': {
            'voxels': lidar.features,
            'coors': lidar.indices,
            'spatial_shape': lidar.spatial_shape,
            'batch_size': lidar.batch_size,
        },
        'targets': labels,
    }

    model.train()
    ret_dict, tb_dict, disp_dict = model(batch_dict)
    print(ret_dict['loss'])
    print(tb_dict)
    print(disp_dict)
