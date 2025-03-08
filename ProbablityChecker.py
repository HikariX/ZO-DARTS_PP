import numpy as np
import torch
from count_param import *
import matplotlib.pyplot as plt
import pandas as pd
import count_param
import StructureSampler
import glob

if __name__ == '__main__':
    dataset_list = ['OrganSMNIST', 'OrganAMNIST', 'OrganCMNIST', 'TissueMNIST',
                        'PneumoniaMNIST', 'BreastMNIST', 'OCTMNIST', 'BloodMNIST', 'PathMNIST', 'DermaMNIST']
    method_list = ['DARTSV2', 'MiLeNAS', 'ZO-DARTS', 'ZO-DARTS+']
    
    
    for seed in range(1, 4):
        for method in ['DARTS', 'MileNAS', 'ZO']:
            for dataset in dataset_list:
                path1 = './exps/NAS-Bench-201-algos/Others/{:}_{:}_+{:}/'.format(method, dataset[:-5].lower(), seed)
                path2 = './exps/NAS-Bench-201-algos/Others2/{:}_{:}_+{:}/'.format(method, dataset[:-5].lower(), seed)
                
                print(dataset, glob.glob(path1 + "*.log")[0].split('/')[-1], glob.glob(path2 + "*.log")[0].split('/')[-1])
                prob1, _, _ = StructureSampler.read_prob(path1)
                prob2, _, _ = StructureSampler.read_prob(path2)
                
                print(prob1, prob2)
                if np.sum(prob1 - prob2) != 0:
                    print('Wrong! Method {:} in dataset {:} and seed {:} is vibrating!'.format(method, dataset[:-5].lower(), seed))
                else:
                    print('Same!')
