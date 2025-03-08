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


class TinyNetworkSPARSEZOMIXEXITGumbel(nn.Module):
    def __init__(
            self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats
    ):
        super(TinyNetworkSPARSEZOMIXEXITGumbel, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
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
            1e-3 * torch.randn(num_edge, len(search_space))
        )
        self.arch_parameters_mix = None
        self.hidden_list = []
        self.arch_parameters_exit = None
        self.buffer_exit = 0

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def initialize_mixed_op(self):
        self.arch_parameters_mix = nn.Parameter(
            1e-3 * torch.randn(6, 3).cuda()
        )  # Initialize the proposed weight for mixed size search. We only consider 7x7, 5x5, and 3x3.
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                cell.initialize_mixed_op()

    def initialize_exits(self):
        self.arch_parameters_exit = nn.Parameter(
            1e-3 * torch.randn(1, 3).cuda()
        )

    def get_alphas_list(self):
        if self.arch_parameters_mix is None:
            return [self.arch_parameters]
        else:
            return [self.arch_parameters, self.arch_parameters_mix, self.arch_parameters_exit]

    def show_alphas(self):
        with torch.no_grad():
            if self.arch_parameters_mix is None:
                return "arch-parameters :\n{:}".format(
                    # nn.functional.softmax(self.arch_parameters, dim=-1).cpu()
                    nn.functional.gumbel_softmax(self.arch_parameters, tau=0.5, dim=-1).cpu()
                )
            else:
                return "arch-parameters :\n{:}, arch-parameters-mix :\n{:}, arch-parameters-exit :\n{:}".format(
                    nn.functional.gumbel_softmax(self.arch_parameters, tau=0.5, dim=-1).cpu(),
                    nn.functional.gumbel_softmax(self.arch_parameters_mix, tau=0.5, dim=-1).cpu(),
                    nn.functional.gumbel_softmax(self.arch_parameters_exit, tau=0.5, dim=-1).cpu()
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

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                    if op_name == 'kernel_variable_conv' and self.arch_parameters_mix is not None:
                        weights_mix = self.arch_parameters_mix[self.edge2index[node_str]]
                        mix_op_names = ['nor_conv_7x7', 'nor_conv_5x5', 'nor_conv_3x3']
                        op_name = mix_op_names[weights_mix.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def forward(self, inputs):
        # alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        alphas = nn.functional.gumbel_softmax(self.arch_parameters, tau=0.5, dim=-1).cpu()
        feature = self.stem(inputs)
        is_edge = True
        if self.arch_parameters_exit is None:
            for i, cell in enumerate(self.cells):
                if isinstance(cell, SearchCell):
                    feature = cell(feature, alphas, self.arch_parameters_mix)
                else:
                    feature = cell(feature)
        else:
            exit_weight = nn.functional.gumbel_softmax(self.arch_parameters_exit, tau=0.5, dim=-1).cpu()
            feature = self.cells[0](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[1](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[2](feature, alphas, self.arch_parameters_mix)
            temp1 = feature * exit_weight[0, 0]
            feature = self.cells[3](feature, alphas, self.arch_parameters_mix)
            temp2 = feature * exit_weight[0, 1]
            feature = self.cells[4](feature, alphas, self.arch_parameters_mix)
            temp3 = feature * exit_weight[0, 2]

            feature = self.cells[5](temp1 + temp2 + temp3)

            feature = self.cells[6](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[7](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[8](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[9](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[10](feature, alphas, self.arch_parameters_mix)

            feature = self.cells[11](feature)

            feature = self.cells[12](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[13](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[14](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[15](feature, alphas, self.arch_parameters_mix)
            feature = self.cells[16](feature, alphas, self.arch_parameters_mix)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits
