##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, random, torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ..cell_operations import OPS


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201RetrainCell(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        structure,
        affine=False,
        track_running_stats=True,
        params=None,
        index=None
    ):
        super(NAS201RetrainCell, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        self.index = index
        self.edge_list = ["{:}<-{:}".format(i, j) for i in range(1, max_nodes) for j in range(i)]
        self.structure_list = self.structure_map(structure)
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = self.structure_list[self.edge_map(node_str)]
                if j == 0:
                    op = OPS[op_name](C_in, C_out, stride, affine, track_running_stats)
                else:
                    op = OPS[op_name](C_in, C_out, 1, affine, track_running_stats)
                self.set_params(node_str, op, op_name, params)
                self.edges[node_str] = op
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def structure_map(self, structure):
        # |nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|
        structure_list = structure.split('|')
        structure_list = [i[:-2] for i in structure_list if len(i) > 1]
        return structure_list

    def edge_map(self, node_str):
        return self.edge_list.index(node_str)

    def set_params(self, node_str, op, op_name, params):
        # cells.0.edges.1<-0.2.op.1.weight
        # cells.0.edges.1<-0.2.op.1.bias
        if not params:
            return
        if 'conv_1x1' in op_name:
            key_weight = 'cells.' + str(self.index) + '.edges.' + node_str + '.2.op.1.weight'
            key_bias = 'cells.' + str(self.index) + '.edges.' + node_str + '.2.op.1.bias'
            op.op[1].weight.data = params['search_model'][key_weight].data
            op.op[1].bias.data = params['search_model'][key_bias].data
        elif 'conv_3x3' in op_name:
            key_weight = 'cells.' + str(self.index) + '.edges.' + node_str + '.3.op.1.weight'
            key_bias = 'cells.' + str(self.index) + '.edges.' + node_str + '.3.op.1.bias'
            op.op[1].weight.data = params['search_model'][key_weight].data
            op.op[1].bias.data = params['search_model'][key_bias]
        else:
            pass

    def edge_map(self, node_str):
        return self.edge_list.index(node_str)

    def forward(self, inputs):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                inter_nodes.append(self.edges[node_str](nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]