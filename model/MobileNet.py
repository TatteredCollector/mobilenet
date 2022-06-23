import torch
from torch import nn
from torchsummary import summary


# https://blog.csdn.net/tintinetmilou/article/details/81607721
# Depthwise Conv 深度可分离卷积 简单来说
# 就是在每个输入通道上使用一个卷积进行运算和输出
# 输出通道数与输入通道数相等
# Pointwise COnv 卷积核大小为1*1*M*N M为上一层的通道数，N为这一层的输出通道数
# 普通卷积 = DW+PW 极大减少计算量
# 卷积池化取整默认向下 取整


# 深度卷积类 继承Sequential 步距（针对DW层，其他为1）
class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBnRelu, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expend_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expend_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layer = []
        if expend_ratio != 1:
            # 扩张，提高通道数，获得更多特征，不改变形状
            layer.append(ConvBnRelu(in_channel=in_channel, out_channel=hidden_channel,
                                    kernel_size=1))
        # extend()
        # 函数用于在列表末尾一次性追加另一个序列中的多个值
        # （用新列表扩展原来的列表）
        layer.extend([
            # DW卷积 stride = 论文中的s
            ConvBnRelu(in_channel=hidden_channel, out_channel=hidden_channel,
                       stride=stride, groups=hidden_channel),
            # PW 卷积 liner
            nn.Conv2d(in_channels=hidden_channel, out_channels=out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel)])
        self.conv = nn.Sequential(*layer)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 通道薄化
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)
        # print(input_channel,last_channel)
        inverted_residuals_set = [
            # t,c,n,s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        features = []

        features.append(ConvBnRelu(in_channel=3, out_channel=input_channel, stride=2))

        for t, c, n, s in inverted_residuals_set:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_channel=input_channel, out_channel=output_channel, stride=stride, expend_ratio=t))
                input_channel = output_channel
        features.append(ConvBnRelu(in_channel=input_channel, out_channel=last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=num_classes)
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    net = MobileNetV2()
    print(net)
    summary(net, input_size=(3, 224, 224), device='cpu')
