##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
import torch.nn as nn
from sparsemax import Sparsemax
import math

sparsemax = Sparsemax(dim=1)

__all__ = ["OPS", "RAW_OP_CLASSES", "ResNetBasicblock", "SearchSpaceNames"]

OPS = {
    "none": lambda C_in, C_out, stride, affine, track_running_stats: Zero(
        C_in, C_out, stride
    ),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "avg", affine, track_running_stats
    ),
    "max_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(
        C_in, C_out, stride, "max", affine, track_running_stats
    ),
    "kernel_variable_conv": lambda C_in, C_out, stride, affine, track_running_stats: MixedReLUConvBN(
        C_in,
        C_out,
        (7, 7),
        (stride, stride),
        (3, 3),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_7x7": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (7, 7),
        (stride, stride),
        (3, 3),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_5x5": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (2, 2),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(
        C_in,
        C_out,
        (1, 1),
        (stride, stride),
        (0, 0),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (1, 1),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dua_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: DualSepConv(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (2, 2),
        (1, 1),
        affine,
        track_running_stats,
    ),
    "dil_sepc_3x3": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (3, 3),
        (stride, stride),
        (2, 2),
        (2, 2),
        affine,
        track_running_stats,
    ),
    "dil_sepc_5x5": lambda C_in, C_out, stride, affine, track_running_stats: SepConv(
        C_in,
        C_out,
        (5, 5),
        (stride, stride),
        (4, 4),
        (2, 2),
        affine,
        track_running_stats,
    ),
    'squeeze': lambda C_in, C_out, stride, affine, track_running_stats: SqueezeExcitation(
        C_in,
    ),
    'inverted_res': lambda C_in, C_out, stride, affine, track_running_stats: InvertedResidual(
        C_in,
        C_out,
        (3, 3),
        stride
    ),
    "skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: Identity()
    if stride == 1 and C_in == C_out
    else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}

CONNECT_NAS_BENCHMARK = ["none", "skip_connect", "nor_conv_3x3"]
NAS_BENCH_201 = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
Eugenio_NAS_BENCH_201 = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3", "squeeze",
                         "inverted_res"]
NAS_BENCH_201_VARIED = ["none", "skip_connect", "nor_conv_1x1", "kernel_variable_conv", "avg_pool_3x3"]
ENLARGED_NAS_BENCH_201 = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3", "dua_sepc_3x3",
                          "dua_sepc_5x5", "dil_sepc_3x3", "dil_sepc_5x5"]
New_N201_1 = ["none", "skip_connect", "squeeze", "inverted_res"]
New_N201_2 = ["none", "skip_connect", "dua_sepc_3x3", "dua_sepc_5x5", "dil_sepc_3x3", "dil_sepc_5x5"]

DARTS_SPACE = [
    "none",
    "skip_connect",
    "dua_sepc_3x3",
    "dua_sepc_5x5",
    "dil_sepc_3x3",
    "dil_sepc_5x5",
    "avg_pool_3x3",
    "max_pool_3x3",
]

SearchSpaceNames = {
    "connect-nas": CONNECT_NAS_BENCHMARK,
    "nats-bench": NAS_BENCH_201,
    "nas-bench-201": NAS_BENCH_201,
    "darts": DARTS_SPACE,
    "enlarged-nas-bench-201": ENLARGED_NAS_BENCH_201,
    "eugenio-nas-bench-201": Eugenio_NAS_BENCH_201,
    "new-n201-1": New_N201_1,
    "new-n201-2": New_N201_2,
    "nas-bench-201-varied": NAS_BENCH_201_VARIED
}


class Conv2dNormActivation(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, padding=1, activation_layer=None,
                 inplace=True):
        super(Conv2dNormActivation, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(C_out, eps=0.001, momentum=0.01)
        self.activation = activation_layer(inplace=inplace) if activation_layer is not None else nn.Identity()

    def forward(self, x, mix_weight=None):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, C_in, squeeze_factor=3):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_in, C_in // squeeze_factor, 1),
            nn.ReLU(),
            nn.Conv2d(C_in // squeeze_factor, C_in, 1),
            nn.Hardsigmoid()
        )

    def forward(self, x, mix_weight=None):
        scale = self.se(x)
        return x * scale


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, exp_size=1, use_se=None,
                 activation_fn='RE'):
        super(InvertedResidual, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_layer = nn.ReLU if activation_fn == 'RE' else nn.Hardswish
        hidden_dim = int(in_channels * exp_size)

        self.block = nn.Sequential()

        # Expansion phase
        if exp_size != 1:
            self.block.add_module("expansion", Conv2dNormActivation(in_channels, hidden_dim, kernel_size=1,
                                                                    activation_layer=activation_layer))

        # Depthwise convolution
        self.block.add_module("depthwise",
                              Conv2dNormActivation(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                                   padding=padding, groups=hidden_dim,
                                                   activation_layer=activation_layer))

        if use_se:
            # Squeeze-and-Excitation layer
            self.block.add_module("se", SqueezeExcitation(hidden_dim))

        # Projection phase
        self.block.add_module("projection", nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        self.block.add_module("norm", nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01))

    def forward(self, x, mix_weight=None):
        identity = x if self.use_residual else None
        x = self.block(x)
        if identity is not None:
            x += identity
        return x


class ReLUConvBN(nn.Module):
    def __init__(
            self,
            C_in,
            C_out,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x, mix_weight=None):
        return self.op(x)


class MixedReLUConvBN(nn.Module):
    def __init__(
            self,
            C_in,
            C_out,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats=True,
    ):
        super(MixedReLUConvBN, self).__init__()
        self.op_list = nn.ModuleList()
        self.kernel_size_list = [(7, 7), (5, 5), (3, 3)]
        self.padding_list = [(3, 3), (2, 2), (1, 1)]
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.argument_list = [C_in, C_out, stride, dilation]
        op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                self.kernel_size_list[0],
                stride=stride,
                padding=self.padding_list[0],
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=self.track_running_stats
            ),
        )
        self.op_list.append(op)

    def initialize_mixed_weight(self):
        # Initialize mixed weight with only weight of the center inherieted
        #        print("In initialize_mixed_weight of cell_operations.py. If gradients can't be transferred, check here plz")
        for i in range(1, 3):
            op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    self.argument_list[0],
                    self.argument_list[1],
                    self.kernel_size_list[i],
                    stride=self.argument_list[2],
                    padding=self.padding_list[i],
                    dilation=self.argument_list[3],
                    bias=not self.affine,
                ),
                nn.BatchNorm2d(
                    self.argument_list[1], affine=self.affine, track_running_stats=self.track_running_stats
                ),
            )
            self.op_list.append(op.cuda())

        self.op_list[1][1].weight.data.copy_(self.op_list[0][1].weight.data[:, :, 1:6, 1:6])
        self.op_list[2][1].weight.data.copy_(self.op_list[0][1].weight.data[:, :, 2:5, 2:5])
        if not self.affine:
            self.op_list[1][1].bias.data.copy_(self.op_list[0][1].bias.data)
            self.op_list[2][1].bias.data.copy_(self.op_list[0][1].bias.data)

    def forward(self, x, mix_weight=None):
        if mix_weight is None:
            return self.op_list[0](x)  # In the beginning, only search 7x7 cells.
        else:
            #            print('In cell_operations', mix_weight)
            return sum(layer(x) * w for layer, w in zip(self.op_list, mix_weight))


class SepConv(nn.Module):
    def __init__(
            self,
            C_in,
            C_out,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats=True,
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=not affine),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x, mix_weight=None):
        return self.op(x)


class DualSepConv(nn.Module):
    def __init__(
            self,
            C_in,
            C_out,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats=True,
    ):
        super(DualSepConv, self).__init__()
        self.op_a = SepConv(
            C_in,
            C_in,
            kernel_size,
            stride,
            padding,
            dilation,
            affine,
            track_running_stats,
        )
        self.op_b = SepConv(
            C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats
        )

    def forward(self, x, mix_weight=None):
        x = self.op_a(x)
        x = self.op_b(x)
        return x


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(
            inplanes, planes, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes, planes, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__
        )
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Module):
    def __init__(
            self, C_in, C_out, stride, mode, affine=True, track_running_stats=True
    ):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def forward(self, inputs, mix_weight=None):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, mix_weight=None):
        return x


class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x, mix_weight=None):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(
                        C_in, C_outs[i], 1, stride=stride, padding=0, bias=not affine
                    )
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(
                C_in, C_out, 1, stride=stride, padding=0, bias=not affine
            )
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(
            C_out, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x, mix_weight=None):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


# Auto-ReID: Searching for a Part-Aware ConvNet for Person Re-Identification, ICCV 2019
class PartAwareOp(nn.Module):
    def __init__(self, C_in, C_out, stride, part=4):
        super().__init__()
        self.part = 4
        self.hidden = C_in // 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv_list = nn.ModuleList()
        for i in range(self.part):
            self.local_conv_list.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(C_in, self.hidden, 1),
                    nn.BatchNorm2d(self.hidden, affine=True),
                )
            )
        self.W_K = nn.Linear(self.hidden, self.hidden)
        self.W_Q = nn.Linear(self.hidden, self.hidden)

        if stride == 2:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 2)
        elif stride == 1:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 1)
        else:
            raise ValueError("Invalid Stride : {:}".format(stride))

    def forward(self, x):
        batch, C, H, W = x.size()
        assert H >= self.part, "input size too small : {:} vs {:}".format(
            x.shape, self.part
        )
        IHs = [0]
        for i in range(self.part):
            IHs.append(min(H, int((i + 1) * (float(H) / self.part))))
        local_feat_list = []
        for i in range(self.part):
            feature = x[:, :, IHs[i]: IHs[i + 1], :]
            xfeax = self.avg_pool(feature)
            xfea = self.local_conv_list[i](xfeax)
            local_feat_list.append(xfea)
        part_feature = torch.cat(local_feat_list, dim=2).view(batch, -1, self.part)
        part_feature = part_feature.transpose(1, 2).contiguous()
        part_K = self.W_K(part_feature)
        part_Q = self.W_Q(part_feature).transpose(1, 2).contiguous()
        weight_att = torch.bmm(part_K, part_Q)
        attention = torch.softmax(weight_att, dim=2)
        aggreateF = torch.bmm(attention, part_feature).transpose(1, 2).contiguous()
        features = []
        for i in range(self.part):
            feature = aggreateF[:, :, i: i + 1].expand(
                batch, self.hidden, IHs[i + 1] - IHs[i]
            )
            feature = feature.view(batch, self.hidden, IHs[i + 1] - IHs[i], 1)
            features.append(feature)
        features = torch.cat(features, dim=2).expand(batch, self.hidden, H, W)
        final_fea = torch.cat((x, features), dim=1)
        outputs = self.last(final_fea)
        return outputs


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = x.new_zeros(x.size(0), 1, 1, 1)
        mask = mask.bernoulli_(keep_prob)
        x = torch.div(x, keep_prob)
        x.mul_(mask)
    return x


# Searching for A Robust Neural Architecture in Four GPU Hours
class GDAS_Reduction_Cell(nn.Module):
    def __init__(
            self, C_prev_prev, C_prev, C, reduction_prev, affine, track_running_stats
    ):
        super(GDAS_Reduction_Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(
                C_prev_prev, C, 2, affine, track_running_stats
            )
        else:
            self.preprocess0 = ReLUConvBN(
                C_prev_prev, C, 1, 1, 0, 1, affine, track_running_stats
            )
        self.preprocess1 = ReLUConvBN(
            C_prev, C, 1, 1, 0, 1, affine, track_running_stats
        )

        self.reduction = True
        self.ops1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        C,
                        C,
                        (1, 3),
                        stride=(1, 2),
                        padding=(0, 1),
                        groups=8,
                        bias=not affine,
                    ),
                    nn.Conv2d(
                        C,
                        C,
                        (3, 1),
                        stride=(2, 1),
                        padding=(1, 0),
                        groups=8,
                        bias=not affine,
                    ),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C, C, 1, stride=1, padding=0, bias=not affine),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                ),
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        C,
                        C,
                        (1, 3),
                        stride=(1, 2),
                        padding=(0, 1),
                        groups=8,
                        bias=not affine,
                    ),
                    nn.Conv2d(
                        C,
                        C,
                        (3, 1),
                        stride=(2, 1),
                        padding=(1, 0),
                        groups=8,
                        bias=not affine,
                    ),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(C, C, 1, stride=1, padding=0, bias=not affine),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                ),
            ]
        )

        self.ops2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                ),
                nn.Sequential(
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.BatchNorm2d(
                        C, affine=affine, track_running_stats=track_running_stats
                    ),
                ),
            ]
        )

    @property
    def multiplier(self):
        return 4

    def forward(self, s0, s1, drop_prob=-1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        X0 = self.ops1[0](s0)
        X1 = self.ops1[1](s1)
        if self.training and drop_prob > 0.0:
            X0, X1 = drop_path(X0, drop_prob), drop_path(X1, drop_prob)

        # X2 = self.ops2[0] (X0+X1)
        X2 = self.ops2[0](s0)
        X3 = self.ops2[1](s1)
        if self.training and drop_prob > 0.0:
            X2, X3 = drop_path(X2, drop_prob), drop_path(X3, drop_prob)
        return torch.cat([X0, X1, X2, X3], dim=1)


# To manage the useful classes in this file.
RAW_OP_CLASSES = {"gdas_reduction": GDAS_Reduction_Cell}
