# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple


def rescale_2x(x: torch.Tensor, scale_factor=2):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def global_avgpool(x: torch.Tensor):
    return x.mean((-1, -2), keepdim=True)


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError


class ConvBN(nn.Module):
    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, g=1, bias=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        return self.bn(self.conv(x))


class CBR(nn.Module):
    def __init__(self, in_dim, out_dim, k, s=1, p=0, d=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class ConvGNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        gn_groups=8,
        bias=False,
        act_name="relu",
        inplace=True,
    ):
        """
        执行流程Conv2d => GroupNormalization [=> Activation]

        Args:
            in_planes: 模块输入通道数
            out_planes: 模块输出通道数
            kernel_size: 内部卷积操作的卷积核大小
            stride: 卷积步长
            padding: 卷积padding
            dilation: 卷积的扩张率
            groups: 卷积分组数，需满足pytorch自身要求
            gn_groups: GroupNormalization的分组数，默认为4
            bias: 是否启用卷积的偏置，默认为False
            act_name: 使用的激活函数，默认为relu，设置为None的时候则不使用激活函数
            inplace: 设置激活函数的inplace参数
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="gn", module=nn.GroupNorm(num_groups=gn_groups, num_channels=out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name, inplace=inplace))


class PixelNormalizer(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.

        Args:
            mean (tuple, optional): the mean value. Defaults to (0.485, 0.456, 0.406).
            std (tuple, optional): the std value. Defaults to (0.229, 0.224, 0.225).
        """
        super().__init__()
        # self.norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        self.register_buffer(name="mean", tensor=torch.Tensor(mean).reshape(3, 1, 1))
        self.register_buffer(name="std", tensor=torch.Tensor(std).reshape(3, 1, 1))

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean.flatten()}, std={self.std.flatten()})"

    def forward(self, x):
        """normalize x by the mean and std values

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor

        Albumentations:

        ```
            mean = np.array(mean, dtype=np.float32)
            mean *= max_pixel_value
            std = np.array(std, dtype=np.float32)
            std *= max_pixel_value
            denominator = np.reciprocal(std, dtype=np.float32)

            img = img.astype(np.float32)
            img -= mean
            img *= denominator
        ```
        """
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x


class LayerNorm2d(nn.Module):
    """
    From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
