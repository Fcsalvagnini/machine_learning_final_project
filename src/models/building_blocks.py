import torch
from torch import nn
from typing import Union

class Conv3DBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1,
            padding: Union[int, tuple] = 0, padding_mode: str = "zeros",
            bias: bool = True, normalization: nn.Module = None,
            activation: nn.Module = None, upsampling: nn.Module = None
    ) -> None:
        super().__init__()
        self.conv3d_layer = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias, padding_mode=padding_mode
        )

        self.norm_layer = normalization
        self.activation = activation
        self.upsampling = upsampling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d_layer(x)

        if self.norm_layer:
            x = self.norm_layer(x)

        if self.activation:
            x = self.activation(x)

        not_upsampled = x
        if self.upsampling:
            x = self.upsampling(x)

        return x, not_upsampled

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu'
                )

                if m.bias is not None:
                    m.bias.data.zero_()
