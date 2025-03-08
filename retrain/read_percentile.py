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
    if len(result) != 300:
        print('Wrong! File {:} did not have enough values!'.format(path))
    
    return max(result), structure, exit

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

    resource_constraint = [1, 2, 3]

    # jump_list = ['DermaMNIST', 'BloodMNIST']
    jump_list = []
    dataset_list = ['PathMNIST', 'BloodMNIST', 'DermaMNIST', 'PneumoniaMNIST', 'TissueMNIST', 'OCTMNIST', 'BreastMNIST', 'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST']
    for constraint in resource_constraint:
        for dataset in dataset_list:
            if dataset in jump_list:
                continue
            result_dict[dataset][constraint] = [] # Prepare a recorder
            size_dict[dataset][constraint] = [] # Prepare a recorder
            for seed in range(1, 4):
                for r in range(1, 4):
                    path = './result_percentile/{:}_constraint{:}_seed{:}_round{:}.log'.format(dataset[:-5].lower(), str(constraint), str(seed), str(r))
                    max_result, structure, cell = read_result(path)
                    result_dict[dataset][constraint].append(max_result)
                    size_dict[dataset][constraint].append(count_param.resource_calculator(structure, cell, num_class=class_dict[dataset]) / 1000000)
            mean = round(statistics.mean(result_dict[dataset][constraint]), 1)
            std = round(statistics.stdev(result_dict[dataset][constraint]), 2)
            result_dict[dataset][constraint] = [mean, std]
            
            mean = round(statistics.mean(size_dict[dataset][constraint]), 2)
            std = round(statistics.stdev(size_dict[dataset][constraint]), 1)
            size_dict[dataset][constraint] = [mean, std]
            print(dataset, constraint, result_dict[dataset][constraint], size_dict[dataset][constraint])
   

