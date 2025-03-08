import torch
import numpy as np
import glob, os, sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from count_param import *
import time

op_list = ["none", "skip_connect", "nor_conv_1x1", "kernel_variable_conv", "avg_pool_3x3"]
op_list_old = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
mix_list = ['nor_conv_7x7', 'nor_conv_5x5', 'nor_conv_3x3']
exit_list = [1, 2, 3]

class_dict = {'OrganSMNIST':11,
              'OrganCMNIST': 11,
              'OrganAMNIST': 11,
              'OCTMNIST': 4,
              'PneumoniaMNIST': 2,
              'BreastMNIST': 2,
              'BloodMNIST': 8,
              'TissueMNIST': 8,
              'DermaMNIST': 7,
              'PathMNIST': 9}

def read_time(path):
    file = open(glob.glob(path + "*.log")[0], "r")
    content = file.readlines()

    if 'ZO_SA' in path: # Both ZO_SAP and ZO_SA can be find.
        for line in content:
            if '[039-050]' in line and 'time-cost=' in line:
                _time = line.split('time-cost=')[1][:-2]
                break
    else:
        for line in content:
            if 'last-geno is' in line:
                _time = line.split('cost ')[1].split(' s, last-geno is ')[0]

    return float(_time)

def read_prob(path):
    file = open(glob.glob(path + "*.log")[0], "r")

    # 打开并读取每个文件的内容
    content = file.readlines()
    arch_prob_list = []
    mix_prob_list = []
    exit_prob_list = []
    counter_dict = {'arch':0, 'mix':0, 'exit':0}
    flag_start = 0
    flag_dict = {'arch':0, 'mix':0, 'exit':0}
    if 'ZO_SAP' in path:
        for line in content:
            if 'The 039-050-th epoch :' in line:
                flag_start = 1
                continue

            if flag_start and 'arch-parameters :' in line:
                flag_dict['arch'] = 1
                continue  # Skip an unnecessary line.
            elif flag_start and not flag_dict['arch'] and 'tensor' in line:
                if not mix_prob_list:
                    flag_dict['mix'] = 1
                else:
                    flag_dict['exit'] = 1

            if flag_dict['arch']:
                arch_prob_list.append(line)
                counter_dict['arch'] += 1
                if counter_dict['arch'] == 18:
                    flag_dict['arch'] = 0
            elif flag_dict['mix']:
                mix_prob_list.append(line)
                counter_dict['mix'] += 1
                if counter_dict['mix'] == 18:
                    flag_dict['mix'] = 0
            elif flag_dict['exit']:
                exit_prob_list.append(line)
                counter_dict['exit'] += 1
                if counter_dict['exit'] == 3:
                    flag_dict['exit'] = 0
                    break
        arch = text2prob(arch_prob_list)
        mix = text2prob(mix_prob_list)
        exit = text2prob(exit_prob_list)
    elif 'ZO_SA' in path: # For ZO_SA.
        for line in content:
            if 'The 039-050-th epoch :' in line:
                flag_start = 1
                continue

            if flag_start and 'arch-parameters :' in line:
                flag_dict['arch'] = 1
                continue  # Skip an unnecessary line.

            if flag_dict['arch']:
                arch_prob_list.append(line)
                counter_dict['arch'] += 1
                if counter_dict['arch'] == 6:
                    break
        arch = text2prob(arch_prob_list)
        mix = None
        exit = None
    else:
        for line in content:
            if 'The 049-050-th epoch :' in line:
                flag_start = 1
                continue

            if flag_start and 'arch-parameters :' in line:
                flag_dict['arch'] = 1
                continue  # Skip an unnecessary line.

            if flag_dict['arch']:
                arch_prob_list.append(line)
                counter_dict['arch'] += 1
                if counter_dict['arch'] == 6:
                    break
        arch = text2prob(arch_prob_list)
        mix = None
        exit = None

    return arch, mix, exit

def text2prob(input):
    input[0] = input[0][8:] # The first line contains unwanted words.
    input[-1] = input[-1].split('])')[0]
    input = [eval(i.split(',\n')[0].strip()) for i in input]

    prob_list = []
    # Normalize
    for i in range(len(input)):
        prob_list.append(input[i] / np.sum(input[i]))

    return np.array(prob_list)

def structure_generator(arch, mix, exit):
    # np.random.seed(int(time.time()))
    arch_list = [np.random.choice(op_list, p=(arch[i])) for i in range(arch.shape[0])]
    arch_list = [arch_list[i] if arch_list[i] != 'kernel_variable_conv' else np.random.choice(mix_list, p=mix[i]) for i
                 in range(arch.shape[0])]
    cell_list = [np.random.choice(exit_list, p=(exit[i])) for i in range(3)]

    structure_list = []
    edge_idx = [0, 0, 1, 0, 1, 2]
    for i in range(3):
        structure_str = '|'
        for j in range(6*i, 6*(i+1)):
            structure_str += arch_list[j] + '~{:}|'.format(str(edge_idx[j - 6*i]))
            if j - 6*i in [0, 2]:
                structure_str += '+|' # Add extra symbols, let it resemble the original structure form
        structure_list.append(structure_str)

    return structure_list, cell_list

def structure_generator_old(arch):
    # np.random.seed(int(time.time()))
    arch_list = [np.random.choice(op_list_old, p=(arch[i])) for i in range(arch.shape[0])]

    edge_idx = [0, 0, 1, 0, 1, 2]

    structure_str = '|'
    for i in range(6):
        structure_str += arch_list[i] + '~{:}|'.format(str(edge_idx[i]))
        if i in [0, 2]:
            structure_str += '+|'  # Add extra symbols, let it resemble the original structure form

    return structure_str

def structure_filter(arch, mix, exit, dataset, limit):
    while True:
        structure_list, cell_list = structure_generator(arch, mix, exit)
        consumption = resource_calculator(structure_list, cell_list, num_class=class_dict[dataset])
        if consumption < limit[1] and consumption > limit[0]: # Satisfies both upper and lower bounds.
            return structure_list, cell_list

def budget_generate():
    dataset_list = ['OrganSMNIST', 'OrganAMNIST', 'OrganCMNIST', 'TissueMNIST',
                    'PneumoniaMNIST', 'BreastMNIST', 'OCTMNIST', 'BloodMNIST', 'PathMNIST', 'DermaMNIST']
    np.random.seed(1)
    data_dict = {}
    p02_list = []
    p46_list = []
    p8X_list = []
    for dataset in dataset_list:
        data_dict[dataset] = []
        for seed in range(1, 4):
            path = './exps/NAS-Bench-201-algos/noPenaltyFull/ZO_SAP_{:}_+{:}/'.format(dataset[:-5].lower(), str(seed))
            arch, mix, exit = read_prob(path)
            for i in range(300):
                structure_list, cell_list = structure_generator(arch, mix, exit)
                data_dict[dataset].append(resource_calculator(structure_list, cell_list, num_class=class_dict[dataset]))

        p02_list.append([dataset, 0, np.percentile(data_dict[dataset], 20)])
        p46_list.append([dataset, np.percentile(data_dict[dataset], 40), np.percentile(data_dict[dataset], 60)])
        p8X_list.append([dataset, np.percentile(data_dict[dataset], 80), np.percentile(data_dict[dataset], 95)])
        # print([dataset, 0, np.percentile(data_dict[dataset], 20)])
        # print([dataset, np.percentile(data_dict[dataset], 80), np.percentile(data_dict[dataset], 95)])

    f1 = open('size_p02.txt', 'w')
    f2 = open('size_p46.txt', 'w')
    f3 = open('size_p8X.txt', 'w')

    for i in range(10):
        f1.write(str(p02_list[i]) + '\n')
        f2.write(str(p46_list[i]) + '\n')
        f3.write(str(p8X_list[i]) + '\n')

    print(p02_list)
    print(p46_list)
    print(p8X_list)

def plot():
    budget = 1000000
    data_list = []
    dataset_list = ['OrganSMNIST', 'OrganAMNIST', 'OrganCMNIST', 'TissueMNIST',
                    'PneumoniaMNIST', 'BreastMNIST', 'OCTMNIST', 'BloodMNIST', 'PathMNIST', 'DermaMNIST']
    for dataset in dataset_list:
        for seed in range(1, 4):
            path = './exps/NAS-Bench-201-algos/noPenaltyFull/ZO_SAP_{:}_+{:}/'.format(dataset[:-5].lower(), str(seed))
            arch, mix, exit = read_prob(path)
            for i in range(300):
                structure_list, cell_list = structure_generator(arch, mix, exit)
                data_list.append(
                    [resource_calculator(structure_list, cell_list, num_class=class_dict[dataset]), dataset])

    df = pd.DataFrame(data_list, columns=['Consumption', 'Dataset'])

    for dataset in dataset_list:
        for seed in range(1, 4):
            print(dataset, seed, budget, np.percentile(df[df['Dataset'] == dataset]['Consumption'].values, q=25))
    sns.violinplot(x="Dataset", y="Consumption", data=df, width=1)
    plt.title("Model size distribution at 40th epoch (XL)")
    plt.show()

def sampler_withLimit(dataset):
    budget_dict = {'small': {}, 'medium': {}, 'large': {}}

    file_dict = {'small': 'size_p02.txt',
                 'medium': 'size_p46.txt',
                 'large': 'size_p8X.txt'}
    for size in ['small', 'medium', 'large']:
        content = open('./' + file_dict[size], 'r').readlines()
        for line in content:
            result = eval(line)
            budget_dict[size][result[0]] = result[1:]

    budget_L = budget_dict['small'][dataset][0]
    budget_U = budget_dict['small'][dataset][1]

    budget = 100000

    for seed in range(1, 4):
        if not budget:
            path = "./exps/NAS-Bench-201-algos/noPenaltyFull/ZO_SAP_{:}_+{:}/".format(dataset[:-5].lower(), str(seed))
        else:
            path = "./exps/NAS-Bench-201-algos/Penalty15/ZO_SAP_{:}_+{:}_{:}/".format(dataset[:-5].lower(), str(seed), str(budget))
        arch, mix, exit = read_prob(path)
        for i in range(10):
            structure_list, cell_list = structure_filter(arch, mix, exit, dataset, [budget_L, budget_U])
            print(structure_list, cell_list,
                  resource_calculator(structure_list, cell_list, num_class=class_dict[dataset]))
        print('*' * 100)

def sampler_withoutLimit(_method, dataset):
    for seed in range(1, 4):
        method = _method.split('-')[0]
        path = './exps/NAS-Bench-201-algos/Others/{:}_{:}_+{:}/'.format(method, dataset[:-5].lower(), seed)
        # print(path)
        arch, _, _ = read_prob(path)

        for i in range(3):
            structure = structure_generator_old(arch)
            # print(_method, dataset, structure)
if __name__ == '__main__':
    # budget_generate()
    for method in ['ZO-DARTS', 'DARTS', 'MileNAS']:
        for dataset in class_dict.keys():
            if dataset not in ['OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST']:
                continue
            else:
                sampler_withoutLimit(method, dataset)