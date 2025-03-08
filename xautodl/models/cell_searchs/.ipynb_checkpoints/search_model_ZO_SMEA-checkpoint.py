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

class TinyNetworkZO_SMEA(nn.Module):
    def __init__(
            self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats
    ):
        super(TinyNetworkZO_SMEA, self).__init__()
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
            1e-3 * torch.randn(num_edge * 3, len(search_space))  # For three stages
        )
        self.arch_parameters_mix = None
        self.hidden_list = []
        self.arch_parameters_exit = None
        self.temperature = 1.5
        self.num_class = num_classes

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def initialize_mixed_op(self):
        self.arch_parameters_mix = nn.Parameter(
            1e-3 * torch.randn(18, 3).cuda()
        )  # Initialize the proposed weight for mixed size search. We only consider 7x7, 5x5, and 3x3.
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                cell.initialize_mixed_op()

    def initialize_exits(self):
        self.arch_parameters_exit = nn.Parameter(
            1e-3 * torch.randn(3, 3).cuda()
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
                    sparsemax(self.arch_parameters / self.temperature, dim=-1).cpu()  # 确定输出的数值是否可以.cpu()处理
                )
            else:
                return "arch-parameters :\n{:}, arch-parameters-mix :\n{:}, arch-parameters-exit :\n{:}".format(
                    sparsemax(self.arch_parameters / self.temperature, dim=-1).cpu(),
                    sparsemax(self.arch_parameters_mix / self.temperature, dim=-1).cpu(),
                    sparsemax(self.arch_parameters_exit / self.temperature, dim=-1).cpu()
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
        genotypes = [[], [], []]
        for stage in range(3):
            arch_param = self.arch_parameters[stage * 6:(stage+1) * 6, :]
            if self.arch_parameters_mix is not None:
                arch_param_mix = self.arch_parameters_mix[stage * 6:(stage+1) * 6, :]
            for i in range(1, self.max_nodes):
                xlist = []
                for j in range(i):
                    node_str = "{:}<-{:}".format(i, j)
                    with torch.no_grad():
                        weights = arch_param[self.edge2index[node_str]]
                        op_name = self.op_names[weights.argmax().item()]
                        if op_name == 'kernel_variable_conv' and self.arch_parameters_mix is not None:
                            if self.arch_parameters_mix is not None:
                                weights_mix = arch_param_mix[self.edge2index[node_str]]
                                mix_op_names = ['nor_conv_7x7', 'nor_conv_5x5', 'nor_conv_3x3']
                                op_name = mix_op_names[weights_mix.argmax().item()]
                            else:
                                op_name = 'nor_conv_7x7'
                    xlist.append((op_name, j))
                genotypes[stage].append(tuple(xlist))
        return [Structure(genotypes[0]), Structure(genotypes[1]), Structure(genotypes[2])]

    def forward(self, inputs):
        alphas_list = [sparsemax(self.arch_parameters[:6, :] / self.temperature),
                       sparsemax(self.arch_parameters[6:12, :] / self.temperature),
                       sparsemax(self.arch_parameters[12:, :] / self.temperature)]
        # alphas = sparsemax(self.arch_parameters / self.temperature)
        if self.arch_parameters_mix is None:
            alphas_mix_list = self.arch_parameters_mix
        else:# Make sure that the mixed_kernel weights are processed
            alphas_mix_list = [sparsemax(self.arch_parameters_mix[:6, :] / self.temperature),
                       sparsemax(self.arch_parameters_mix[6:12, :] / self.temperature),
                       sparsemax(self.arch_parameters_mix[12:, :] / self.temperature)]
        feature = self.stem(inputs)
        flag = 0

        if self.arch_parameters_exit is None: # Not activated, still for the largest model.
            for i, cell in enumerate(self.cells):
                if isinstance(cell, SearchCell):
                    if i < 3:
                        flag = 0
                    elif i < 7:
                        flag = 1
                    else:
                        flag = 2
                    feature = cell(feature, alphas_list[flag], alphas_mix_list)
                else:
                    feature = cell(feature)
        else:
            exit_weight = sparsemax(self.arch_parameters_exit / self.temperature)
            temp_list = []
            for i in range(3):
                feature = self.cells[i](feature, alphas_list[0], alphas_mix_list[0])
                temp_list.append(feature * exit_weight[0, i])
            feature = self.cells[3](temp_list[0] + temp_list[1] + temp_list[2])

            temp_list = []
            for i in range(3):
                feature = self.cells[i + 4](feature, alphas_list[1], alphas_mix_list[1])
                temp_list.append(feature * exit_weight[1, i])
            feature = self.cells[7](temp_list[0] + temp_list[1] + temp_list[2])

            temp_list = []
            for i in range(3):
                feature = self.cells[i + 8](feature, alphas_list[2], alphas_mix_list[2])
                temp_list.append(feature * exit_weight[2, i])
            feature = temp_list[0] + temp_list[1] + temp_list[2]

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits