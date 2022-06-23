# python中@staticmethod方法，类似于C++中的static，
# 方便将外部函数集成到类体中，
# 主要是可以在不实例化类的情况下直接访问该方法，
# 如果你去掉staticmethod,
# 在方法中加self也可以通过实例化访问方法也是可以集成。

# Python 为弱类型语言，可以调用typing 进行数据类型强调和注释
# https://docs.python.org/zh-cn/3/library/typing.html

from typing import Callable, Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from functools import partial
from torchsummary import summary


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    # Typing.Optional类
    # 可选类型，作用几乎和带默认值的参数等价，
    # 不同的是使用Optional会告诉你的IDE或者框架：
    # 这个参数除了给定的默认值外还可以是None
    # Callable [..., ]
    # 第一个类型（int）代表参数类型
    # 第二个类型（str）代表返回值类型
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                      stride=stride, groups=groups, padding=padding,
                      bias=False),
            norm_layer(num_features=out_planes),
            activation_layer(inplace=True)
        )


# SE 通道注意力 https://blog.csdn.net/sll_0909/article/details/107898830
# 简单描述就是 全局池化-> 全连接层(C/squeeze_factor)->全连接层(C)->与输入相乘
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(in_channels=input_c, out_channels=squeeze_c, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=squeeze_c, out_channels=input_c, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        # hardsigmoid计算量比sigmoid小
        scale = F.hardsigmoid(scale, inplace=True)
        return x * scale


class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel_size: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel_size
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(self, config: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if config.stride not in [1, 2]:
            raise ValueError("illegal stride value")

        self.use_res_connect = (config.stride == 1 and config.input_c == config.out_c)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if config.use_hs else nn.ReLU
        # 扩展
        if config.expanded_c != config.input_c:
            layers.append(ConvBNActivation(in_planes=config.input_c, out_planes=config.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))
        # DW 卷积
        layers.append(ConvBNActivation(in_planes=config.expanded_c, out_planes=config.expanded_c,
                                       kernel_size=config.kernel, stride=config.stride,
                                       groups=config.expanded_c, norm_layer=norm_layer,
                                       activation_layer=activation_layer))
        if config.use_se:
            layers.append(SqueezeExcitation(input_c=config.expanded_c))

        # PW卷积
        # nn.identity模块不改变输入。 相当于占位符
        layers.append(ConvBNActivation(in_planes=config.expanded_c, out_planes=config.out_c,
                                       kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))
        self.bneck = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        result = self.bneck(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        # 判断输出参数是否符合要求
        if not inverted_residual_setting:
            raise ValueError("基本单元配置不能为空")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise ValueError("配置表单中有不是InvertedResidualConfig的对象")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            # partial()
            # https://blog.csdn.net/qq_33688922/article/details/91890142
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        layers: List[nn.Module] = []

        # 第一层卷积：
        first_conv_out_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(in_planes=3, out_planes=first_conv_out_c,
                                       stride=2, kernel_size=3, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # 骨干层
        for cnf in inverted_residual_setting:
            layers.append(InvertedResidual(config=cnf, norm_layer=norm_layer))

        # last conv
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_out_c = last_conv_input_c * 6
        last_conv = ConvBNActivation(in_planes=last_conv_input_c, out_planes=last_conv_out_c, kernel_size=1, stride=1,
                                     norm_layer=norm_layer, activation_layer=nn.Hardswish)
        layers.append(last_conv)
        self.features = nn.Sequential(*layers)
        self.avrag_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=last_conv_out_c, out_features=last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avrag_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v3_large(num_classes: int = 1000, reduced_tail: bool = False) -> MobileNetV3:
    """

    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    :param num_classes:
    :param reduced_tail:
    :return:
    """
    width_multi = 1.0
    benck_config = partial(InvertedResidualConfig, width_multi=width_multi)

    reduced_divider = 2 if reduced_tail else 1
    adjust_channel = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    inverted_residual_setting = [
        # inputc,kernel_size exp_size out_c se Nl stride
        benck_config(16, 3, 16, 16, False, 'RE', 1),
        benck_config(16, 3, 64, 24, False, 'RE', 2),
        benck_config(24, 3, 72, 24, False, 'RE', 1),
        benck_config(24, 5, 72, 40, True, 'RE', 2),
        benck_config(40, 5, 120, 40, True, 'RE', 1),
        benck_config(40, 5, 120, 40, True, 'RE', 1),
        benck_config(40, 3, 240, 80, False, 'HS', 2),
        benck_config(80, 3, 200, 80, False, 'HS', 1),
        benck_config(80, 3, 184, 80, False, 'HS', 1),
        benck_config(80, 3, 184, 80, False, 'HS', 1),
        benck_config(80, 3, 480, 112, True, 'HS', 1),
        benck_config(112, 3, 672, 112, True, 'HS', 1),
        benck_config(112, 5, 672, 160 // reduced_divider, True, 'HS', 2),
        benck_config(160 // reduced_divider, 5, 960 // reduced_divider, 160 // reduced_divider, True, 'HS', 1),
        benck_config(160 // reduced_divider, 5, 960 // reduced_divider, 160 // reduced_divider, True, 'HS', 1),

    ]
    last_channel = adjust_channel(1280 // reduced_divider)

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel, num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000, reduced_tail: bool = False) -> MobileNetV3:
    """
        weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

    :param num_classes:
    :param reduced_tail:
    :return:
    """
    width_multi = 1.0
    bencek_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduced_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        bencek_conf(16, 3, 16, 16, True, "RE", 2),
        bencek_conf(16, 3, 72, 24, False, "RE", 2),
        bencek_conf(24, 3, 88, 24, False, "RE", 1),
        bencek_conf(24, 5, 96, 40, True, "HS", 2),
        bencek_conf(40, 5, 240, 40, True, "HS", 1),
        bencek_conf(40, 5, 240, 40, True, "HS", 1),
        bencek_conf(40, 5, 120, 48, True, "HS", 1),
        bencek_conf(48, 5, 144, 48, True, "HS", 1),
        bencek_conf(48, 5, 288, 96 // reduced_divider, True, "HS", 2),
        bencek_conf(96 // reduced_divider, 5, 576, 96 // reduced_divider, True, "HS", 1),
        bencek_conf(96 // reduced_divider, 5, 576, 96 // reduced_divider, True, "HS", 1),

    ]

    last_channel = adjust_channels(1024 // reduced_divider)
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel, num_classes=num_classes)


if __name__ == "__main__":
    net = mobilenet_v3_small()

    summary(net, input_size=(3, 224, 224), device='cpu')
