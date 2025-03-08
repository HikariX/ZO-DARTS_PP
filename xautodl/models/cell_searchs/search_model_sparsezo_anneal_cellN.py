##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
import torch
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells import NAS201SearchCell as SearchCell
from .genotypes import Structure
from sparsemax import Sparsemax

sparsemax = Sparsemax(dim=1)

class TinyNetworkSPARSEZOANNEALCELLN(nn.Module):
    def __init__(
        self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats
    ):
        super(TinyNetworkSPARSEZOANNEALCELLN, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
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
        self.arch_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge * 3, len(search_space)) # For three stages
        )
        self.arch_parameters_cell = None
        self.temperature = 1.5
        self.cell_temperature = 1

    def initialize_archparam_cells(self):
        self.arch_parameters_cell = nn.Parameter(
            1e-3 * torch.randn(3, 3).cuda()
        )

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas_list(self):
        if self.arch_parameters_cell is None:
            return [self.arch_parameters]
        else:
            return [self.arch_parameters, self.arch_parameters_cell]

    def show_alphas(self):
        with torch.no_grad():
            if self.arch_parameters_cell is None:
                return "arch-parameters :\n{:}\n{:}\n{:}".format(
                    sparsemax(self.arch_parameters / self.temperature).cpu(),
                )
            else:
                return "arch-parameters :\n{:}, arch-parameters-cell :\n{:}".format(
                        sparsemax(self.arch_parameters / self.temperature).cpu(),
                        sparsemax(self.arch_parameters_cell / self.cell_temperature).cpu()
                    )

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    # Modified for 3 stages
    def genotype(self):
        genotypes_list = []
        for stage in range(3):
            genotypes = []
            for i in range(1, self.max_nodes):
                xlist = []
                for j in range(i):
                    node_str = "{:}<-{:}".format(i, j)
                    with torch.no_grad():
                        weights = self.arch_parameters[self.edge2index[node_str] * stage]
                        op_name = self.op_names[weights.argmax().item()]
                    xlist.append((op_name, j))
                genotypes.append(tuple(xlist))
            genotypes_list.append(Structure(genotypes))
        return genotypes_list

    def forward(self, inputs):
        alphas_list = [sparsemax(self.arch_parameters[:6, :] / self.temperature),
                       sparsemax(self.arch_parameters[6:12, :] / self.temperature),
                       sparsemax(self.arch_parameters[12:, :] / self.temperature)]

        feature = self.stem(inputs)
        flag = 0

        if self.arch_parameters_cell is None:
            for i, cell in enumerate(self.cells):
                if isinstance(cell, SearchCell):
                    if i < 5:
                        flag = 0
                    elif i < 12:
                        flag = 1
                    else:
                        flag = 2
                    feature = cell(feature, alphas_list[flag], weightss_mix=None)
                else:
                    feature = cell(feature)
        else:
            exit_weight = sparsemax(self.arch_parameters_cell / self.cell_temperature)

            temp_list = []
            for i in range(5):
                feature = self.cells[i](feature, alphas_list[0], weightss_mix=None)
                if i >= 2:
                    temp_list.append(feature * exit_weight[0, i - 2])
            feature = self.cells[5](temp_list[0] + temp_list[1] + temp_list[2])

            temp_list = []
            for i in range(5):
                feature = self.cells[i + 6](feature, alphas_list[1], weightss_mix=None)
                if i >= 2:
                    temp_list.append(feature * exit_weight[1, i - 2])
            feature = self.cells[11](temp_list[0] + temp_list[1] + temp_list[2])

            temp_list = []
            for i in range(5):
                feature = self.cells[i + 12](feature, alphas_list[1], weightss_mix=None)
                if i >= 2:
                    temp_list.append(feature * exit_weight[2, i - 2])
            feature = temp_list[0] + temp_list[1] + temp_list[2]

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits
