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

class DiscreteNetworkSPARSEZOANNEALCELLN(nn.Module):
    def __init__(
        self, C, N, max_nodes, num_classes, search_space, structure, affine, track_running_stats, cell_list=None, params=None
    ):
        super(DiscreteNetworkSPARSEZOANNEALCELLN, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        if cell_list is None:
            layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
            layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        else:
            layer_channels = [C] * cell_list[0] + [C * 2] + [C * 2] * cell_list[1] + [C * 4] + [C * 4] * cell_list[2]
            layer_reductions = [False] * cell_list[0] + [True] + [False] * cell_list[1] + [True] + [False] * cell_list[2]

        if params:
            self.stem[0].weight.data = params['search_model']['stem.0.weight'].data
        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        stage = 0
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
                stage += 1
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
                    structure[stage],
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
