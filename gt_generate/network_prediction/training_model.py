import torch
import torch.nn as nn


class SmallConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_tensor):
        return self.block(input_tensor)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SmallConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input_tensor):
        feat = self.conv(input_tensor)
        return feat, self.pool(feat)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = SmallConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, input_tensor, skip_tensor):
        output_tensor = self.up(input_tensor)
        if output_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            output_tensor = torch.nn.functional.interpolate(output_tensor, size=skip_tensor.shape[-2:], mode="bilinear", align_corners=False)
        output_tensor = torch.cat([output_tensor, skip_tensor], dim=1)
        return self.conv(output_tensor)


class TerrainOnlyNet(nn.Module):
    def __init__(self, input_channel_num, hidden_channels):
        super().__init__()
        mid_channels = int(hidden_channels)
        deep_channels = int(hidden_channels * 2)
        bottleneck_channels = int(hidden_channels * 4)
        self.enc1 = DownBlock(input_channel_num, mid_channels)
        self.enc2 = DownBlock(mid_channels, deep_channels)
        self.bottleneck = SmallConvBlock(deep_channels, bottleneck_channels)
        self.dec2 = UpBlock(bottleneck_channels, deep_channels, deep_channels)
        self.dec1 = UpBlock(deep_channels, mid_channels, mid_channels)
        self.terrain_head = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, input_tensor):
        skip1, down1 = self.enc1(input_tensor)
        skip2, down2 = self.enc2(down1)
        bottleneck = self.bottleneck(down2)
        up2 = self.dec2(bottleneck, skip2)
        feat = self.dec1(up2, skip1)
        return {"terrain_z_inpainted": self.terrain_head(feat)}
