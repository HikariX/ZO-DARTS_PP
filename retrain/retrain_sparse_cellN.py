import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from xautodl.utils import get_model_infos
from xautodl.models.cell_searchs.retrain_model_cellN import DiscreteNetworkSPARSEZOANNEALCELLN
from xautodl.datasets import get_datasets
from cutmix_transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
from xautodl.config_utils import load_config
import os, sys, time, glob, argparse

batchsize_dict = {'OrganAMNIST': 128,
                  'OrganCMNIST': 128,
                  'OrganSMNIST': 128,
                  'PneumoniaMNIST': 16,
                  'OCTMNIST': 128,
                  'BreastMNIST': 64,
                  'BloodMNIST': 32,
                  'DermaMNIST': 32,
                  'PathMNIST': 128,
                  'TissueMNIST': 256
                  }

class_num_dict = {'OrganSMNIST':11,
              'OrganCMNIST': 11,
              'OrganAMNIST': 11,
              'OCTMNIST': 4,
              'PneumoniaMNIST': 2,
              'BreastMNIST': 2,
              'BloodMNIST': 8,
              'TissueMNIST': 8,
              'DermaMNIST': 7,
              'PathMNIST': 9}

def structure_reader(structure_recorder):
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
    path = "../exps/NAS-Bench-201-algos/Penalty15/ZO_SAP_" + dataset[:-5].lower() + "_+"  + str(seed) + '_' + str(budget) + '/'

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

        # if "The 039-050-th epoch" in line:
        #     structure_dict['040'] = structure_reader(line)
        #     num_epoch.append('040')
        #     flag0 = 1
        # elif "The 044-050-th epoch" in line:
        #     structure_dict['045'] = structure_reader(line)
        #     num_epoch.append('045')
        #     flag0 = 1
        # elif "The 049-050-th epoch" in line:
        #     structure_dict['050'] = structure_reader(line)
        #     num_epoch.append('050')
        #     flag0 = 1
        if "The 049-050-th epoch" in line:
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


def main(xargs):
    batch_size = batchsize_dict[xargs.dataset]
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 3e-4
    report_freq = 50
    mixup_alpha = 0.1
    cutmix_alpha = 0.1
    num_classes = class_num_dict[xargs.dataset]
    workers = 8
    lr_stepsize = 20
    lr_gamma = 0.1
    lr_warmup_epochs = 10
    lr_warmup_method = 'linear'
    lr_warmup_decay = 0.01
    lr_min = 0
    epochs = 500
    label_smoothing = 0.1
    structure, cell_list = read_structure(xargs.rand_seed, xargs.dataset, xargs.budget)
    structure = structure['050']
    cell_list = cell_list['050']

    print('batch_size:{:}, learning_rate:{:}, momentum:{:}, weight_decay:{:}, '
          'report_freq:{:}, mixup_alpha:{:},cutmix_alpha:{:}, num_classes:{:}, '
          'workers:{:}, dataset:{:}, lr_stepsize:{:}, lr_gamma:{:}, lr_warmup_epochs:{:}, '
          'lr_warmup_method:{:}, lr_warmup_decay:{:}, lr_min:{:}, epochs:{:}, label_smoothing:{:}, structure:{:}, cellnumber:{:}'.format(
        batch_size, learning_rate, momentum, weight_decay, report_freq, mixup_alpha,
        cutmix_alpha, num_classes, workers, xargs.dataset, lr_stepsize, lr_gamma, lr_warmup_epochs,
        lr_warmup_method, lr_warmup_decay, lr_min, epochs, label_smoothing, structure, cell_list))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # params = torch.load('SPARSEZOAnneal+2+0.8/checkpoint/seed-2-basic.pth')
    params = None

    # structure = ['|dil_sepc_3x3~0|+|dua_sepc_5x5~0|dil_sepc_5x5~1|+|skip_connect~0|dil_sepc_5x5~1|skip_connect~2|',
    #              '|dua_sepc_3x3~0|+|dua_sepc_3x3~0|dua_sepc_3x3~1|+|skip_connect~0|dua_sepc_5x5~1|dua_sepc_5x5~2|',
    #              '|dil_sepc_5x5~0|+|none~0|dua_sepc_5x5~1|+|dua_sepc_5x5~0|none~1|dua_sepc_3x3~2|']

    net = DiscreteNetworkSPARSEZOANNEALCELLN(C=16, N=3, max_nodes=4, num_classes=num_classes,
                                             search_space='nas-bench-201',
                                             structure=structure, affine=False, track_running_stats=False,
                                             cell_list=cell_list, params=params)
    net.to(device)

    # # 加载CIFAR-10数据集
    # train_data, test_data, xshape, class_num = get_datasets(
    #     "cifar10", "/root/autodl-fs/ZO_DARTS_light_test/nasbench201/dataset/cifar", -1
    # )
    datapath = "../nasbench201/dataset/" + xargs.dataset.lower() + ".npz"
    train_data, test_data, xshape, class_num = get_datasets(
        xargs.dataset, datapath, -1, mode='retrain_nopipe'
    )

    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, num_categories=num_classes
    )

    if mixup_cutmix is not None:
        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), learning_rate, momentum, weight_decay)
    # flop, param = get_model_infos(net, xshape)



    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - lr_warmup_epochs, eta_min=lr_min
    )

    if lr_warmup_epochs > 0:
        if lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        elif lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=lr_warmup_decay, total_iters=lr_warmup_epochs
            )
        else:
            # raise RuntimeError(f "Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported.")
            pass
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                                 schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                                                                 milestones=[lr_warmup_epochs])
    else:
        lr_scheduler = main_lr_scheduler

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
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120", "OCTMNIST", "PneumoniaMNIST", "BreastMNIST", "TissueMNIST",
                 "OrganAMNIST", "OrganCMNIST", "OrganSMNIST", "PathMNIST", "DermaMNIST", "BloodMNIST"],
        default="OrganSMNIST",
    )
    parser.add_argument("--rand_seed", type=int, default=2, help="manual seed")
    parser.add_argument("--budget", type=int, default=1000000, help='resource budget')
    args = parser.parse_args()
    main(args)

