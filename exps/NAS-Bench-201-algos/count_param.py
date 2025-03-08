import torch
import numpy as np
import glob, os, sys

def structure_reader(structure_recorder):
    # print(structure_recorder)
    structures = structure_recorder.split('Structure(4 nodes with ')[1:]
    structures[0] = structures[0][:-3]  # Delete excessive brackets
    structures[1] = structures[1][:-3]
    structures[2] = structures[2][:-4]
    return structures

def exit_reader(exit_recorder):
    exit_recorder = exit_recorder[1:4]
    exit_recorder[0] = exit_recorder[0][8:]
    exit_recorder[-1] = exit_recorder[-1][:-3]
    exit_recorder = [eval(i.split(',\n')[0].strip()) for i in exit_recorder]
    cell_list = [i.index(max(i)) + 1 for i in exit_recorder]
    return cell_list


def read_structure(seed=None, dataset=None, budget=None):
    # path = "./Penalty15/ZO_SAP_" + dataset[:-5].lower() + "_+"  + str(seed) + "_" + str(budget) + '/'
    path = "./Cifar/ZO_SAP_+" + str(seed) + "_" + str(budget) + '/'

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

def structure_reader_old(structure_recorder):
    structures = structure_recorder.split('Structure(4 nodes with ')[1][:-3]
    # print(structures)
    return structures

def read_structure_old(method=None, seed=None, dataset=None):
    path = './Cifar/' + method + '_+' + str(seed) + '/'
    file = open(glob.glob(path + "*.log")[0], "r")

    content = file.readlines()

    for line in content:
        if 'last-geno is Structure' in line:
            structure = structure_reader_old(line)
            break
    return structure

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
            # print('Input & output:', input_channel, output_channel)
            # print(element, kernel * kernel * input_channel * output_channel + output_channel)
            param += kernel * kernel * input_channel * output_channel + output_channel
    return param

def resource_calculator(structure, cell_list=None, budget=200000, num_class=11):
    param = 0
    in_C = [16, 32, 64]
    out_C = [16, 32, 64]
    if not cell_list:
        stage = 0
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
            stage_param_num *= cell_list[stage]
            param += stage_param_num
    param = param + 16 * 3 * 3 + 16 * 2  # Add head, Conv + BN
    param = param + count_reduc_num(16, 32) + count_reduc_num(32, 64)  # Add reduction cells
    param = param + 64 * 2  # Last_act, the last activation, contains one BN layer.
    param = param + 64 * num_class + num_class  # Add classifier

    # print(param)
    # print(param - budget, budget + 100000 - param)
    return param

# Breast
#loose
# structure = ['|avg_pool_3x3~0|+|avg_pool_3x3~0|nor_conv_7x7~1|+|nor_conv_7x7~0|nor_conv_1x1~1|skip_connect~2|)',
#              '|skip_connect~0|+|none~0|nor_conv_7x7~1|+|avg_pool_3x3~0|avg_pool_3x3~1|none~2|',
#              '|none~0|+|nor_conv_1x1~0|avg_pool_3x3~1|+|nor_conv_7x7~0|skip_connect~1|avg_pool_3x3~2|']
#
# cell_list = [2, 1, 1]

#tight
# structure = ['|none~0|+|avg_pool_3x3~0|skip_connect~1|+|nor_conv_3x3~0|nor_conv_1x1~1|nor_conv_5x5~2|',
#            '|skip_connect~0|+|avg_pool_3x3~0|avg_pool_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|skip_connect~2|',
#            '|avg_pool_3x3~0|+|skip_connect~0|none~1|+|avg_pool_3x3~0|nor_conv_1x1~1|skip_connect~2|']
# cell_list = [2, 2, 2]

# #OCT
# structure = ['|none~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+|avg_pool_3x3~0|none~1|skip_connect~2|',
#              '|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|avg_pool_3x3~0|nor_conv_7x7~1|nor_conv_1x1~2|',
#              '|nor_conv_3x3~0|+|none~0|nor_conv_1x1~1|+|avg_pool_3x3~0|nor_conv_1x1~1|none~2|']
# cell_list = [1, 1, 1]
#
# structure = ['|skip_connect~0|+|none~0|skip_connect~1|+|nor_conv_5x5~0|nor_conv_1x1~1|nor_conv_5x5~2|)',
#  '|nor_conv_1x1~0|+|nor_conv_1x1~0|none~1|+|skip_connect~0|nor_conv_1x1~1|skip_connect~2|',
#  '|skip_connect~0|+|none~0|nor_conv_7x7~1|+|avg_pool_3x3~0|nor_conv_1x1~1|nor_conv_5x5~2|']
# cell_list = [2, 1, 1]

# Pneumonia
# tight
# structure = ['|none~0|+|nor_conv_1x1~0|skip_connect~1|+|skip_connect~0|avg_pool_3x3~1|nor_conv_1x1~2|',
#              '|skip_connect~0|+|skip_connect~0|nor_conv_3x3~1|+|avg_pool_3x3~0|nor_conv_3x3~1|nor_conv_1x1~2|',
#              '|none~0|+|nor_conv_3x3~0|skip_connect~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_3x3~2|']
# cell_list = [2, 1, 3]

# # loose
# structure = ['|none~0|+|nor_conv_3x3~0|skip_connect~1|+|avg_pool_3x3~0|skip_connect~1|avg_pool_3x3~2|',
#              '|avg_pool_3x3~0|+|skip_connect~0|nor_conv_7x7~1|+|avg_pool_3x3~0|nor_conv_7x7~1|nor_conv_7x7~2|',
#              '|none~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|nor_conv_7x7~0|nor_conv_7x7~1|skip_connect~2|']
# cell_list = [2, 2, 1]

# structure = ['|skip_connect~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|avg_pool_3x3~0|nor_conv_7x7~1|nor_conv_3x3~2|',
#              '|none~0|+|avg_pool_3x3~0|none~1|+|nor_conv_1x1~0|none~1|skip_connect~2|',
#              '|none~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+|nor_conv_7x7~0|none~1|nor_conv_1x1~2|']
# cell_list = [2, 3, 2]

# structure = ['|skip_connect~0|+|skip_connect~0|skip_connect~1|+|nor_conv_1x1~0|nor_conv_3x3~1|avg_pool_3x3~2|',
#              '|none~0|+|none~0|skip_connect~1|+|avg_pool_3x3~0|avg_pool_3x3~1|skip_connect~2|',
#              '|avg_pool_3x3~0|+|nor_conv_5x5~0|none~1|+|nor_conv_1x1~0|nor_conv_5x5~1|nor_conv_1x1~2|']
# cell_list = [3, 2, 3]


# structure = '[Structure(4 nodes with |nor_conv_1x1~0|+|nor_conv_1x1~0|skip_connect~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_7x7~2|), Structure(4 nodes with |none~0|+|nor_conv_1x1~0|avg_pool_3x3~1|+|none~0|nor_conv_1x1~1|avg_pool_3x3~2|), Structure(4 nodes with |none~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|avg_pool_3x3~0|nor_conv_7x7~1|skip_connect~2|)]'
# cell_list = [2, 2, 3]
# budget_list = [i * 100000 for i in range(2, 12, 2)]

# for seed in range(1, 5):
#     print('Seed:', seed)
#     for budget in budget_list:
#         structures, cell_list = read_structure(seed, 'OrganSMNIST', budget)
#         print('Budget:', budget)
#         resource_calculator(structures, cell_list)
# structure = read_structure(structure)
# resource_calculator(structure, cell_list, budget=1000000, num_class=4)

class_dict = {'OrganSMNIST':11,
              'OrganCMNIST': 11,
              'OrganAMNIST': 11,
              'OCTMNIST': 4,
              'PneumoniaMNIST': 2,
              'BreastMNIST': 2,
              'BloodMNIST': 8,
              'TissueMNIST': 8,
              'DermaMNIST': 7,
              'PathMNIST': 9,
              'CIFAR-10': 10}
budget_list = [100000, 200000, 1000000]

for budget in budget_list:
    for seed in range(1, 4):
        for dataset in ['CIFAR-10']:
        # for dataset in ['OrganSMNIST', 'OrganAMNIST', 'OrganCMNIST', 'TissueMNIST', 'PneumoniaMNIST', 'BreastMNIST', 'OCTMNIST', 'BloodMNIST', 'PathMNIST', 'DermaMNIST']: 
            structure_dict, exit_dict = read_structure(seed, dataset, budget)
            consumption = [resource_calculator(structure_dict[epoch], exit_dict[epoch], num_class=class_dict[dataset]) for epoch in ['040', '045', '050']]
            print('Dataset: ', dataset, 'Budget: ', budget, 'Seed: ', seed, 'Consumption: ', consumption[2])
            # with open('./structure_list_seed' + str(seed) + '.txt', 'a+', encoding='utf-8') as file:
            #     file.write('budget:%s\n' % str(budget))
            #     for item in structures:
            #         file.write("%s\n" % item)
            #     for item in cell_list:
            #         file.write("%s\n" % item)
for method in ['MiLeNAS', 'DARTS', 'ZO']:
    for seed in range(1, 4):
        structure_dict = [read_structure_old(method, seed)]
        consumption = resource_calculator(structure_dict, num_class=10)
        print('Method:', method, 'Dataset: ', 'CIFAR-10', 'Seed: ', seed, 'Consumption: ', consumption)
        # print(structure_dict)
        # with open('./structure_list_seed' + str(seed) + '.txt', 'a+', encoding='utf-8') as file:
        #     file.write('budget:%s\n' % str(budget))
        #     for item in structures:
        #         file.write("%s\n" % item)
        #     for item in cell_list:
        #         file.write("%s\n" % item)
        
