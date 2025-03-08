import torch
import numpy as np
import glob, os, sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
def structure_reader(structure_recorder, isDifferent=False):
    structures = structure_recorder.split('Structure(4 nodes with ')[1:]
    if isDifferent:
        structures[0] = structures[0][:-3]  # Delete excessive brackets
        structures[1] = structures[1][:-3]
        structures[2] = structures[2][:-4]
    else:
        structures[0] = structures[0][:-3]  # Delete excessive brackets
    return structures

def exit_reader(exit_recorder):
    exit_recorder = exit_recorder[1:4]
    exit_recorder[0] = exit_recorder[0][8:]
    exit_recorder[-1] = exit_recorder[-1][:-3]
    exit_recorder = [eval(i.split(',\n')[0].strip()) for i in exit_recorder]
    cell_list = [i.index(max(i)) + 1 for i in exit_recorder]
    return cell_list

def structure_reader_old(structure_recorder):
    structures = structure_recorder.split('structure:')[1][:-1]
    return structures

def read_structure_old(seed=None, method=None, dataset=None):
    method_dict = {'DARTSV2':'_darts', 'MiLeNAS': '_m', 'ZO-DARTS': '_zo', 'ZO-DARTS+': ''}
    path = './retrain/result_autodl/seed' + str(seed) + '/' + method + '/' + dataset[:-5].lower() + method_dict[method] + '.log'
    file = open(path, 'r')

    content = file.readlines()

    for line in content:
        if 'structure' in line:
            structure = structure_reader_old(line)
            break
    return [structure]

def read_structure(seed=None, dataset=None, budget=None):
    if not budget:
        path = "./exps/NAS-Bench-201-algos/noPenaltyFull/ZO_SAP_" + dataset[:-5].lower() + "_+"  + str(seed) + '/'
    else:
        path = "./exps/NAS-Bench-201-algos/Penalty15/ZO_SAP_" + dataset[:-5].lower() + "_+" + str(seed) + '_' + str(budget) + '/'
    file = open(glob.glob(path + "*.log")[0], "r")

    # 打开并读取每个文件的内容
    content = file.readlines()
    exit_recorder = []
    flag0 = 0
    flag1 = 0
    structure_dict = {}
    exit_dict = {}
    counter = 0
    num_epoch = []
    for line in content:
        if 'find the highest validation accuracy' in line:
            continue

        if "The 039-050-th epoch" in line:
            structure_dict['040'] = structure_reader(line)
            num_epoch.append('040')
            flag0 = 1
        elif "The 044-050-th epoch" in line:
            structure_dict['045'] = structure_reader(line)
            num_epoch.append('045')
            flag0 = 1
        elif "The 049-050-th epoch" in line:
            structure_dict['050'] = structure_reader(line)
            num_epoch.append('050')
            flag0 = 1

        if flag0 and "arch-parameters-exit :" in line:
            flag1 = 1

        if flag0 and flag1:
            exit_recorder.append(line)
            counter += 1

        if counter == 4: # Refresh the recorder
            exit_dict[num_epoch[-1]] = exit_reader(exit_recorder)
            flag0 = 0
            flag1 = 0
            counter = 0
            exit_recorder = []

    return structure_dict, exit_dict

def read_structure_ZO_plus(seed=None, dataset=None, budget=None):
    path = "./exps/NAS-Bench-201-algos/Penalty15_ZO+/ZO_SA_" + dataset[:-5].lower() + "_+"  + str(seed) + '_' + str(budget) + '/'
    isDifferent = False
    file = open(glob.glob(path + "*.log")[0], "r")

    # 打开并读取每个文件的内容
    content = file.readlines()
    structure_dict = {}
    for line in content:
        if 'find the highest validation accuracy' in line:
            continue

        if "The 039-050-th epoch" in line:
            structure_dict['040'] = structure_reader(line, isDifferent)
        elif "The 044-050-th epoch" in line:
            structure_dict['045'] = structure_reader(line, isDifferent)
        elif "The 049-050-th epoch" in line:
            structure_dict['050'] = structure_reader(line, isDifferent)

    return structure_dict, None

def count_reduc_num(input_channel, output_channel):
    num = (input_channel + output_channel) * output_channel * 3 * 3
    num += input_channel * output_channel
    num += output_channel * 4  # 2 BatchNorm for conv_a and conv_b, each contains 2 sets of params.
    return num

def count_param_num(structure, input_channel, output_channel):
    param = 0
    for element in structure:
        if element[:3] == 'nor':
            kernel = int(element[-1])
            # print(element, kernel * kernel * input_channel * output_channel + output_channel)
            param += kernel * kernel * input_channel * output_channel + output_channel
    return param

def resource_calculator(structure, cell_list, num_class=11):
    param = 0
    in_C = [16, 32, 64]
    out_C = [16, 32, 64]
    if not cell_list:
        for stage in range(3):
            stage_param_num = 0
            structure_element = structure[0].split('|')
            structure_element = [i.split('~')[0] for i in structure_element if len(i) > 3]
            stage_param_num += count_param_num(structure_element, in_C[stage], out_C[stage])
            stage_param_num *= 3
            param += stage_param_num
    else:
        for stage in range(3):
            stage_param_num = 0
            structure_element = structure[stage].split('|')
            structure_element = [i.split('~')[0] for i in structure_element if len(i) > 3]
            stage_param_num += count_param_num(structure_element, in_C[stage], out_C[stage])
            # print(structure_element, count_param_num(structure_element, in_C[stage], out_C[stage]))
            stage_param_num *= cell_list[stage]
            param += stage_param_num
    param = param + 16 * 3 * 3 + 16 * 2  # Add head, Conv + BN
    param = param + count_reduc_num(16, 32) + count_reduc_num(32, 64)  # Add reduction cells
    param = param + 64 * 2  # Last_act, the last activation, contains one BN layer.
    param = param + 64 * num_class + num_class  # Add classifier

    # print(param)
    # print(param - budget, budget + 100000 - param)
    return param

# class_dict = {'OrganSMNIST':11,
#               'OrganCMNIST': 11,
#               'OrganAMNIST': 11,
#               'OCTMNIST': 4,
#               'PneumoniaMNIST': 2,
#               'BreastMNIST': 2,
#               'BloodMNIST': 8,
#               'TissueMNIST': 8,
#               'DermaMNIST': 7,
#               'PathMNIST': 9}
# budget_list = [0, 100000, 200000, 1000000]
# data_dict = {}
# result_list = []
# budget_dict = {0:'Origin', 100000:'XS', 200000:'S', 1000000:'L'}

if __name__ == '__main__':
    # for budget in budget_list:
    #     if budget:
    #         key = '050'
    #     else:
    #         key = '040'
    #     for dataset in ['OrganSMNIST', 'OrganAMNIST', 'OrganCMNIST', 'TissueMNIST',
    #                     'PneumoniaMNIST', 'BreastMNIST', 'OCTMNIST', 'BloodMNIST', 'PathMNIST', 'DermaMNIST']:
    #         data_dict[dataset] = []
    #         # For kernel-varied ZO++
    #         for seed in range(1, 4):
    #             structure_dict, exit_dict = read_structure(seed, dataset, budget)
    #             consumption = \
    #             [resource_calculator(structure_dict[epoch], exit_dict[epoch], num_class=class_dict[dataset]) for epoch
    #              in [key]][0]
    #
    #             temp = round(consumption / 1000000, 2)
    #             data_dict[dataset].append(temp)
    #             result_list.append([dataset, temp, budget_dict[budget]])
    #
    #         for seed in range(1, 4):
    #             if not budget:
    #                 break
    #             else:
    #                 structure_dict, exit_dict = read_structure_ZO_plus(seed, dataset, budget)
    #             consumption = \
    #             [resource_calculator(structure_dict[epoch], None, num_class=class_dict[dataset]) for epoch in [key]][0]
    #
    #             temp = round(consumption / 1000000, 2)
    #             data_dict[dataset].append(temp)
    #             result_list.append([dataset, temp, '*' + budget_dict[budget]])
    #
    # df = pd.DataFrame(result_list, columns=['Dataset', 'Num', 'Budget'])
    # grouped = df['Num'].groupby([df['Dataset'], df['Budget']])
    # print(df[df['Dataset'] == 'BloodMNIST'])
    #
    # print(grouped.mean())
    # plt.figure(figsize=(18, 3))
    # ax = sns.barplot(x='Dataset', y='Num', hue='Budget', capsize=0.05, estimator=np.mean, data=df)
    # plt.ylabel('Param Num')
    # plt.title('Parameter comparisons between different constraints')
    # plt.legend(loc='upper right', fontsize='9')
    # plt.savefig('./param.eps', bbox_inches='tight', pad_inches=0.2)
    # plt.show()
    #
    str = '[Structure(4 nodes with |skip_connect~0|+|avg_pool_3x3~0|none~1|+|none~0|skip_connect~1|nor_conv_1x1~2|), Structure(4 nodes with |nor_conv_1x1~0|+|none~0|avg_pool_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_1x1~2|), Structure(4 nodes with |none~0|+|none~0|nor_conv_7x7~1|+|nor_conv_1x1~0|avg_pool_3x3~1|nor_conv_7x7~2|)]'
    structure = structure_reader(str)
    print(structure)
    print(resource_calculator(structure, cell_list=[1,1,2]))