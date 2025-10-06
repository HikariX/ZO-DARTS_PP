import matplotlib.pyplot as plt
import statistics
import os, sys, time, glob, argparse
import math
sys.path.append('..')
import count_param

def read_result_old(path):
    f = open(path, 'r')
    content = f.readlines()
    result = []
    for line in content:
        if 'structure:' in line:
            structure = line.split('structure:')[1].strip()
            
            
        if "Accuracy of the" in line:
            temp = float(line.split(':')[1].split(' %')[0])
            result.append(temp)
        if 'The model learned nothing! Starting a new model.' in line: # Refresh the result.
            result = []
            structure = None

    # print(path, result)
    # print(path, max(result), result.index(max(result)))
    # if len(result) != 300:
    #     print('Wrong! File {:} did not have enough values!'.format(path))
    return max(result), [structure]

def read_result(path):
    f = open(path, 'r')
    content = f.readlines()
    result = []
    structure = []
    exit = []
    for line in content:
        if 'structure:' in line:
            structure = line.split('structure:')[1]
            structure, exit = structure.split(', cellnumber:')
            exit = eval(exit)
            structure = eval(structure)

        if 'The model learned nothing! Starting a new model.' in line: # Refresh the result.
            result = []
        if "Accuracy of the" in line:
            temp = float(line.split(':')[1].split(' %')[0])
            result.append(temp)

    # print(path, result)
    # print(path, max(result), result.index(max(result)))
    # if len(result) != 300:
    #     print('Wrong! File {:} did not have enough values!'.format(path))
    
    return max(result), structure, exit

def parse_architecture_string(s):
    """
    解析结构组字符串，返回操作列表。
    操作列表中的每个元素是元组 (source, target, op_name)
    """
    parts = s.split('+')
    operations = []
    for idx, part in enumerate(parts):
        target_node = idx + 1  # 部分索引对应目标节点编号
        # 去掉部分字符串首尾的"|"，然后分割操作
        part_clean = part.strip('|')
        op_strings = part_clean.split('|')
        for op_str in op_strings:
            if op_str == '':
                continue
            # 解析操作名称和源节点
            if '~' in op_str:
                op_name, source_str = op_str.split('~')
                source_node = int(source_str)
                operations.append((source_node, target_node, op_name))
            else:
                # 如果没有~，可能格式错误，跳过
                continue
    return operations


def find_node_values(operations, num_nodes):
    """
    计算每个节点的值。
    """
    node_values = [0] * num_nodes
    node_values[0] = 1  # 节点0初始值为1

    # 按节点顺序计算节点值
    for target in range(1, num_nodes):
        for source, t, op_name in operations:
            if t != target:
                continue

            if op_name == 'none' or node_values[source] == 0:
                value = 0
            else:
                value = 1  # 未知操作，假设输出1

            # print(source, t, op_name)


            node_values[target] += value
    # print(node_values)
    return node_values


def simplify_operations(operations, num_nodes):
    """
    简化操作：对于每个操作，如果经过多层传递后最终效果为none，则重置为none。
    """
    # 首先计算节点值，这里的思想是如果一个node的全部输入为0，则其输出也应为0。
    node_values = find_node_values(operations, num_nodes)

    # 按节点顺序处理
    for target in range(1, num_nodes):
        if node_values[target] == 0:
            # print(operations)
            for i, op in enumerate(operations):
                source, t, op_name = op
                # print(source, t, op_name)
                if source == target or t == target:
                    operations[i] = (source, t, 'none')
    
    # 其次，如果一个node的全部输出为0，则其输入也应置0。为了方便，此处就手动判定node1和2的情况。
    if operations[-1][-1] == 'none': # 如果2-3是none
        operations[1] = (0, 2, 'none') # 0-2
        operations[2] = (1, 2, 'none') # 1-2
    if operations[2][-1] == 'none' and operations[4][-1] == 'none': # 如果1-2和1-3是none
        operations[0] = (0, 1, 'none') # 0-2

    # print(operations)
    return operations


def generate_architecture_string(operations):
    """
    从操作列表生成结构组字符串。
    """
    # 找出所有目标节点
    targets = sorted(set(target for _, target, _ in operations))

    parts = []
    for target in targets:
        # 找出所有指向该目标节点的操作
        target_ops = [op for op in operations if op[1] == target]

        # 生成该目标节点的操作字符串
        op_strings = []
        for source, _, op_name in target_ops:
            op_strings.append(f"{op_name}~{source}")

        part_str = "|" + "|".join(op_strings) + "|"
        parts.append(part_str)

    return "+".join(parts)

def check_and_simplify_architecture(s):
    """
    主函数：检查结构组并简化。
    """
    operations = parse_architecture_string(s)
    if not operations:
        print("解析失败，无操作。")
        return s

    # 确定节点数量：最大目标节点编号加1
    max_target = max(target for _, target, _ in operations)
    num_nodes = max_target + 1

    # 计算节点值
    node_values = find_node_values(operations, num_nodes)

    alert = None
    # 检查最终输出是否为零
    if node_values[-1] == 0:
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!报警：最终输出为零！该结构组无效。")
        alert = 1
    else:
        # print("最终输出非零，结构有效。")
        alert = 0

    # 简化操作
    simplified_ops = simplify_operations(operations, num_nodes)

    # 生成简化后的结构组字符串
    simplified_str = generate_architecture_string(simplified_ops)
    #
    # print(f"简化前: {s}")
    # print(f"简化后: {simplified_str}")
    return alert, simplified_str

if __name__ == '__main__':
    # dataset_list = ['PathMNIST', 'BloodMNIST', 'DermaMNIST', 'PneumoniaMNIST', 'TissueMNIST', 'OCTMNIST', 'BreastMNIST', 'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST']
    dataset_list = ['PneumoniaMNIST', 'TissueMNIST', 'OCTMNIST', 'BreastMNIST', 'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST']

    jump_list = ['']
    # for method in ['DARTS', 'DARTSAER', 'MileNAS', 'ZO', 'ZOP']:
    #     for dataset in dataset_list:
    #         if dataset in jump_list:
    #             continue
    #         for seed in range(1, 4):
    #             for r in range(1, 4):
    #                 path = './NewExp2025/result_Others/{:}_{:}_seed{:}_round{:}.log'.format(method, dataset[:-5].lower(), str(seed), str(r))
    #                 max_result, structure = read_result_old(path)
    #                 new_structure = check_and_simplify_architecture(structure[0])[1]
    #                 if structure[0] != new_structure:
    #                     print(path)
    #                     print(structure[0])
    #                     print(new_structure)
    resource_constraint = [1, 2, 3]
    for constraint in resource_constraint:
        for dataset in dataset_list:
            if dataset in jump_list:
                continue
            for seed in range(1, 4):
                for r in range(1, 4):
                    path = './NewExp2025/result_percentile/{:}_constraint{:}_seed{:}_round{:}.log'.format(dataset[:-5].lower(), str(constraint), str(seed), str(r))
                    max_result, structure, cell = read_result(path)
                    for s in structure:
                        new_structure = check_and_simplify_architecture(s)[1]
                        if s != new_structure:
                            print(path)
                            print(s)
                            print(new_structure)