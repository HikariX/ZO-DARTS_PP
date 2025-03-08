import numpy as np
import torch
from count_param import *
import matplotlib.pyplot as plt
import pandas as pd
import count_param
import StructureSampler
import glob
import statistics

if __name__ == '__main__':
    # dataset_list = ['PathMNIST', 'BloodMNIST', 'DermaMNIST', 'PneumoniaMNIST', 'TissueMNIST', 'OCTMNIST', 'BreastMNIST', 'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST']
    method_list = ['DARTSV2', 'MiLeNAS', 'ZO-DARTS', 'ZO-DARTS+']
    dataset_list = ['OrganSMNIST', 'OrganCMNIST']
    
    time_dict1 = {'OrganSMNIST':{},
              'OrganCMNIST': {},
              'OrganAMNIST': {},
              'OCTMNIST': {},
              'PneumoniaMNIST': {},
              'BreastMNIST': {},
              'BloodMNIST': {},
              'TissueMNIST': {},
              'DermaMNIST': {},
              'PathMNIST': {}}
    time_dict2 = {'OrganSMNIST':{},
              'OrganCMNIST': {},
              'OrganAMNIST': {},
              'OCTMNIST': {},
              'PneumoniaMNIST': {},
              'BreastMNIST': {},
              'BloodMNIST': {},
              'TissueMNIST': {},
              'DermaMNIST': {},
              'PathMNIST': {}}
    
    for method in ['MileNAS', 'ZO']:
    # for method in ['1', '2', '3']:
        for dataset in dataset_list:
            time_dict1[dataset][method] = []
            time_dict2[dataset][method] = []
            for seed in range(1, 4):
                path1 = './exps/NAS-Bench-201-algos/Others_test/{:}_{:}_+{:}/'.format(method, dataset[:-5].lower(), seed) # True result.
                # path2 = './exps/NAS-Bench-201-algos/Others2/{:}_{:}_+{:}/'.format(method, dataset[:-5].lower(), seed) # Result from different cpu.
                # path1 = './exps/NAS-Bench-201-algos/Penalty15_percentile/ZO_SAP_{:}_+{:}_constraint{:}/'.format(dataset[:-5].lower(), seed, method) # True result.
                # path1 = './exps/NAS-Bench-201-algos/ZO+/ZO_SA_{:}_+{:}/'.format(dataset[:-5].lower(), seed, method) # True result.
                # print(dataset, glob.glob(path1 + "*.log")[0].split('/')[-1], glob.glob(path2 + "*.log")[0].split('/')[-1])
                time_dict1[dataset][method].append(StructureSampler.read_time(path1))
                # print(StructureSampler.read_time(path1), path1)
                # time_dict2[dataset][method].append(StructureSampler.read_time(path2))
                
            mean = round(statistics.mean(time_dict1[dataset][method]), 1)
            std = round(statistics.stdev(time_dict1[dataset][method]), 2)
            time_dict1[dataset][method] = [mean, std]
            
            # mean = round(statistics.mean(time_dict2[dataset][method]), 1)
            # std = round(statistics.stdev(time_dict2[dataset][method]), 2)
            # time_dict2[dataset][method] = [mean, std]
            print(method, dataset, time_dict1[dataset][method])
                
               