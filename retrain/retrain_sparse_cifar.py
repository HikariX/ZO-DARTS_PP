import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from xautodl.utils import get_model_infos
from xautodl.models.cell_searchs.retrain_model_cellN import DiscreteNetworkSPARSEZOANNEALCELLN
from xautodl.models.cell_searchs.retrain_model_darts import DiscreteNetwork
from xautodl.datasets import get_datasets
from cutmix_transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
from xautodl.config_utils import load_config
import os, sys, time, glob, argparse


# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def structure_reader_old(structure_recorder):
    structures = structure_recorder.split('Structure(4 nodes with ')[1][:-3]
    # print(structures)
    return structures

def read_structure_old(method=None, seed=None, dataset=None):
    path = '../exps/NAS-Bench-201-algos/Cifar/' + method + '_+' + str(seed) + '/'
    file = open(glob.glob(path + "*.log")[0], "r")

    content = file.readlines()

    for line in content:
        if 'last-geno is Structure' in line:
            structure = structure_reader_old(line)
            break
    return structure

def read_structure(seed=None, dataset=None, budget=None):
    # path = "../exps/NAS-Bench-201-algos/Cifar/ZO_SAP_+" + str(seed) + "_" + str(budget) + "/"
    path = "../exps/NAS-Bench-201-algos/Cifar/ZO++_+" + str(seed) + "/"
    file = open(glob.glob(path + "*.log")[0], "r")
    # 打开并读取每个文件的内容
    content = file.readlines()
    structure_recorder = ''
    exit_recorder = []
    flag0 = 0
    flag1 = 0
    for line in content:
        if "The 049-050-th epoch" in line:
            flag0 = 1
        if flag0 and "arch-parameters-exit :" in line:
            flag1 = 1

        if "last-geno" in line:
            structure_recorder = line
        elif flag0 and flag1:
            exit_recorder.append(line)
    structures = structure_recorder.split('Structure(4 nodes with ')[1:]
    structures[0] = structures[0][:-3]  # Delete excessive brackets
    structures[1] = structures[1][:-3]
    structures[2] = structures[2][:-4]

    exit_recorder = exit_recorder[1:4]
    exit_recorder[0] = exit_recorder[0][8:]
    exit_recorder[-1] = exit_recorder[-1][:-3]
    exit_recorder = [eval(i.split(',\n')[0].strip()) for i in exit_recorder]
    cell_list = [i.index(max(i)) + 1 for i in exit_recorder]
    return structures, cell_list

def main(xargs):
    batch_size = 256
    _learning_rate = 0.1
    _momentum = 0.9
    _weight_decay = 5e-4
    report_freq = 50
    num_classes = 10
    workers = 8
    lr_min = 0
    epochs = 200
    label_smoothing = 0.1
    dataset = 'CIFAR-10'
    cell_number = 3
    cell_list = None
    
    structure, cell_list = read_structure(xargs.rand_seed, dataset, xargs.budget)
    print('batch_size:{:}, learning_rate:{:}, momentum:{:}, weight_decay:{:}, report_freq:{:}, num_classes:{:}, workers:{:}, dataset:{:}, lr_min:{:}, epochs:{:}, label_smoothing:{:}, structure:{:}, cell_number:{:}.'.format(
        batch_size, _learning_rate, _momentum, _weight_decay, report_freq, num_classes, workers, dataset, lr_min, epochs, label_smoothing, structure, cell_list))
    net = DiscreteNetworkSPARSEZOANNEALCELLN(C=16, N=3, max_nodes=4, num_classes=num_classes,
                                             search_space='nas-bench-201',
                                             structure=structure, affine=False, track_running_stats=False,
                                             cell_list=cell_list, params=None)

    # structure = read_structure_old(xargs.method, xargs.rand_seed, dataset)
    # print('batch_size:{:}, learning_rate:{:}, momentum:{:}, weight_decay:{:}, report_freq:{:}, num_classes:{:}, workers:{:}, dataset:{:}, lr_min:{:}, epochs:{:}, label_smoothing:{:}, structure:{:}, cell_number:{:}.'.format(batch_size, _learning_rate, _momentum, _weight_decay, report_freq, num_classes, workers, dataset, lr_min, epochs, label_smoothing, structure, cell_number))
    # net = DiscreteNetwork(C=16, N=3, max_nodes=4, num_classes=num_classes,
    #                                          search_space='nas-bench-201',
    #                                          structure=structure, affine=False, track_running_stats=False)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # 加载CIFAR-10数据集
    train_data, test_data, xshape, class_num = get_datasets(
        "cifar10", "../nasbench201/dataset/cifar", -1
    )
    
    collate_fn = default_collate

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=_learning_rate, momentum=_momentum, weight_decay=_weight_decay, nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr_min
    )

    
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
        pin_memory=True,
        collate_fn=collate_fn
    )

    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True
    )

    # 训练神经网络模型
    for epoch in range(epochs):  # 遍历数据集多次
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            _, outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % report_freq == report_freq - 1:  # 每2000个小批量数据打印一次损失值
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        lr_scheduler.step()
        # 在测试集上测试神经网络模型
        correct = 0
        total = 0
        # 不进行梯度下降
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                _, outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(correct, total)
        print('Accuracy of the network on the test images: {:} %'.format(
            100.0 * correct / total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("retrain")
    parser.add_argument("--rand_seed", type=int, default=2, help="manual seed")
    parser.add_argument("--method", type=str, default='DARTS', help="method")
    parser.add_argument("--budget", type=int, default=1000000, help='resource budget')
    args = parser.parse_args()
    main(args)