##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
import torch
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .retrain_cells import NAS201RetrainCell as RetrainCell
from .genotypes import Structure

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        self.conv1x1_theta = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                       stride=1, padding=0, bias=False)
        self.conv1x1_phi = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                     stride=1, padding=0, bias=False)
        self.conv1x1_g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.conv1x1_attn = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1,
                                      stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        return x + self.sigma * attn_g

class DiscreteNetworkSPARSEZOANNEALATTENTION(nn.Module):
    def __init__(
        self, C, N, max_nodes, num_classes, search_space, structure, affine, track_running_stats, params=None
    ):
        super(DiscreteNetworkSPARSEZOANNEALATTENTION, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )
        self.attention = SelfAttention(in_channels=self._C)
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        if params:
            self.stem[0].weight.data = params['search_model']['stem.0.weight'].data
        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
                if params:
                    key_weight_a = 'cells.' + str(index) + '.conv_a.op.1.weight'
                    key_weight_b = 'cells.' + str(index) + '.conv_b.op.1.weight'
                    key_weight_down = 'cells.' + str(index) + '.downsample.1.weight'
                    cell.conv_a.op[1].weight.data = params['search_model'][key_weight_a].data
                    cell.conv_b.op[1].weight.data = params['search_model'][key_weight_b].data
                    cell.downsample[1].weight.data = params['search_model'][key_weight_down].data
            else:
                cell = RetrainCell(
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    structure,
                    affine,
                    track_running_stats,
                    params,
                    index
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (
                        num_edge == cell.num_edges and edge2index == cell.edge2index
                    ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        if params:
            self.classifier.weight.data = params['search_model']['classifier.weight'].data
            self.classifier.bias.data = params['search_model']['classifier.bias'].data
        self.structure = structure

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def forward(self, inputs):
        feature = self.stem(inputs)
        feature = self.attention(feature)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, RetrainCell):
                feature = cell(feature)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits
