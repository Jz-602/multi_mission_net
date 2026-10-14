import torch
from torch import nn
import spconv
import spconv.pytorch
from spconv.pytorch import SparseSequential, SparseConv2d


from spconv.core import ConvAlgo


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class SparseConvBlock(spconv.pytorch.SparseModule):
    '''
    Sparse Conv Block
    SparseConv2d for stride > 1 and subMconv2d for stride==1
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, use_subm=True, bias=False):
        super(SparseConvBlock, self).__init__()
        if stride == 1 and use_subm:
            self.conv = spconv.pytorch.SubMConv2d(in_channels, out_channels, kernel_size,
                                                  padding=kernel_size//2, stride=1, bias=bias,)
        else:
            self.conv = spconv.pytorch.SparseConv2d(in_channels, out_channels, kernel_size,
                                                    padding=kernel_size//2, stride=stride, bias=bias)

        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = replace_feature(out, self.norm(out.features))
        out = replace_feature(out, self.act(out.features))

        return out


class SparseBasicBlock(spconv.pytorch.SparseModule):
    '''
    Sparse Conv Block
    '''

    def __init__(self, channels, kernel_size):
        super(SparseBasicBlock, self).__init__()
        self.block1 = SparseConvBlock(channels, channels, kernel_size, 1)
        self.conv2 = spconv.pytorch.SubMConv2d(channels, channels, kernel_size, padding=kernel_size//2,
                                               stride=1, bias=False, algo=ConvAlgo.Native, )
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.act2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act2(out.features))

        return out




class SparseResNet(spconv.pytorch.SparseModule):
    def __init__(
            self,
            layer_nums,
            ds_layer_strides,
            ds_num_filters,
            num_input_features,
            kernel_size=[3, 3, 3, 3]):

        out_channels = ds_num_filters[-1]

        super(SparseResNet, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                kernel_size[i],
                self._layer_strides[i],
                layer_num)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        self.mapping = SparseSequential(
            SparseConv2d(self._num_filters[-1],
                         out_channels, 1, 1, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def _make_layer(self, inplanes, planes, kernel_size, stride, num_blocks):

        layers = []
        layers.append(SparseConvBlock(inplanes, planes,
                      kernel_size=kernel_size, stride=stride, use_subm=False))

        for j in range(num_blocks):
            layers.append(SparseBasicBlock(planes, kernel_size=kernel_size))

        return spconv.pytorch.SparseSequential(*layers)

    def forward(self, pillar_features, coors, input_shape):
        batch_size = len(torch.unique(coors[:, 0]))
        x = spconv.pytorch.SparseConvTensor(
            pillar_features, coors, input_shape, batch_size)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.mapping(x)
        return x.dense()
