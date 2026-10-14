import torch
import torch.nn as nn
import spconv.pytorch as spconv


def PrintSparseState(name, x):
    dense_x = x.dense()
    print(f"\n[{name}]")
    print("features.shape:", tuple(x.features.shape))
    print("indices.shape:", tuple(x.indices.shape))
    print("spatial_shape:", x.spatial_shape)
    print("dense.shape:", tuple(dense_x.shape))
    print("features:")
    print(x.features)
    print("indices:")
    print(x.indices)


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

    x = spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=batch_size,
    )

    return x


def BuildNetwork():
    net = nn.ModuleDict(
        {
            "subm_block": spconv.SparseSequential(
                spconv.SubMConv3d(4, 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(8),
                nn.ReLU(),
            ),
            "down_block": spconv.SparseSequential(
                spconv.SparseConv3d(
                    8,
                    16,
                    kernel_size=3,
                    stride=3,
                    padding=1,
                    bias=False,
                    indice_key="downsample_key",
                ),
                nn.BatchNorm1d(16),
                nn.ReLU(),
            ),
            "up_block": spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    16,
                    8,
                    kernel_size=3,
                    bias=False,
                    indice_key="downsample_key",
                ),
                nn.BatchNorm1d(8),
                nn.ReLU(),
            ),
        }
    )
    return net


def Main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = BuildSparseTensor(device)
    net = BuildNetwork().to(device)
    y_subm = net["subm_block"](x)
    y_down = net["down_block"](y_subm)
    y_up = net["up_block"](y_down)
    dense_y = y_up.dense()

    print("device:", device)
    PrintSparseState("input", x)
    PrintSparseState("after_subm_conv3d", y_subm)
    PrintSparseState("after_sparse_conv3d_stride2", y_down)
    PrintSparseState("after_sparse_inverse_conv3d", y_up)

    print("\n[count_check]")
    print("input active points:", x.features.shape[0])
    print("after downsample active points:", y_down.features.shape[0])
    print("after inverse conv active points:", y_up.features.shape[0])

    assert x.features.shape == (8, 4)
    assert x.indices.shape == (8, 4)
    assert dense_y.dim() == 5
    assert dense_y.shape[0] == 2
    assert dense_y.shape[1] == 8
    assert y_up.features.shape[0] == x.features.shape[0]
    assert torch.equal(y_up.indices, x.indices)

    print("Expanded spconv 3D test with stride-2 downsample and inverse conv passed.")


if __name__ == "__main__":
    Main()
