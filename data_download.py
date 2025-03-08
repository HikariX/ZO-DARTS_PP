import os, sys, time, glob, random, argparse, json
import numpy as np
import numpy as np
random.seed(2501)
from medmnist import PathMNIST, DermaMNIST, PneumoniaMNIST, BreastMNIST, BloodMNIST, OrganCMNIST, OrganSMNIST, OrganAMNIST, TissueMNIST, OCTMNIST


dataset_train = OrganSMNIST(split="train", download=True, size=28)
# dataset_test = OrganSMNIST(split="test", download=True, size=28)
dataset_train = OrganAMNIST(split="train", download=True, size=28)
dataset_train = OrganCMNIST(split="train", download=True, size=28)
dataset_train = PathMNIST(split="train", download=True, size=28)
dataset_train = DermaMNIST(split="train", download=True, size=28)
dataset_train = PneumoniaMNIST(split="train", download=True, size=28)
dataset_train = BreastMNIST(split="train", download=True, size=28)
dataset_train = BloodMNIST(split="train", download=True, size=28)
dataset_train = TissueMNIST(split="train", download=True, size=28)
dataset_train = OCTMNIST(split="train", download=True, size=28)

# data = np.load('/Users/LightningX/.medmnist/organsmnist.npz')
# print(data['test_images'].shape)