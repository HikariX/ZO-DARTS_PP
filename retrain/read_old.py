import matplotlib.pyplot as plt
import statistics
import os, sys, time, glob, argparse
import math
sys.path.append('..')
import count_param
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

def read_result(path):
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

    # print(path, result)
    # print(path, max(result), result.index(max(result)))
    # if len(result) != 300:
    #     print('Wrong! File {:} did not have enough values!'.format(path))
    return max(result), [structure]

if __name__ == '__main__':

    lst = ['path', 'derma', 'blood', 'oct', 'pneumonia', 'breast', 'tissue', 'organa', 'organc', 'organs']

    result_dict = {'OrganSMNIST':{},
              'OrganCMNIST': {},
              'OrganAMNIST': {},
              'OCTMNIST': {},
              'PneumoniaMNIST': {},
              'BreastMNIST': {},
              'BloodMNIST': {},
              'TissueMNIST': {},
              'DermaMNIST': {},
              'PathMNIST': {}}
    
    size_dict = {'OrganSMNIST':{},
              'OrganCMNIST': {},
              'OrganAMNIST': {},
              'OCTMNIST': {},
              'PneumoniaMNIST': {},
              'BreastMNIST': {},
              'BloodMNIST': {},
              'TissueMNIST': {},
              'DermaMNIST': {},
              'PathMNIST': {}}
    
    dataset_list = ['PathMNIST', 'BloodMNIST', 'DermaMNIST', 'PneumoniaMNIST', 'TissueMNIST', 'OCTMNIST', 'BreastMNIST', 'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST']
    # dataset_list = ['PneumoniaMNIST', 'BreastMNIST', 'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST']

    jump_list = ['']
    for method in ['DARTS', 'DARTSAER', 'MileNAS', 'ZO', 'ZOP', 'ZOPP']:
        avg_acc = 0
        avg_size = 0
        for dataset in dataset_list:
            if dataset in jump_list:
                continue
            result_dict[dataset][method] = [] # Prepare a recorder
            size_dict[dataset][method] = [] # Prepare a recorder
            for seed in range(1, 4):
                for r in range(1, 4):
                    path = './NewExp2025/result_Others/{:}_{:}_seed{:}_round{:}.log'.format(method, dataset[:-5].lower(), str(seed), str(r))  
                    max_result, structure = read_result(path)
                    result_dict[dataset][method].append(max_result)
                    size_dict[dataset][method].append(count_param.resource_calculator(structure, None, num_class=class_dict[dataset]) / 1000000)
            mean = round(statistics.mean(result_dict[dataset][method]), 1)
            std = round(statistics.stdev(result_dict[dataset][method]), 2)
            result_dict[dataset][method] = [mean, std]
            avg_acc += mean
            
            mean = round(statistics.mean(size_dict[dataset][method]), 2)
            std = round(statistics.stdev(size_dict[dataset][method]), 1)
            size_dict[dataset][method] = [mean, std]
            avg_size += mean
            print(method, dataset, result_dict[dataset][method], size_dict[dataset][method])
        print(round(avg_acc, 2), round(avg_size, 3))