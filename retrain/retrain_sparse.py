import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from xautodl.utils import get_model_infos
from xautodl.models.cell_searchs.retrain_model import DiscreteNetworkSPARSEZOANNEAL
# from xautodl.models.cell_searchs.retrain_model_attention import DiscreteNetworkSPARSEZOANNEALATTENTION
from xautodl.datasets import get_datasets
from cutmix_transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
from xautodl.config_utils import load_config
import os, sys, time, glob, argparse


# os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

name_list = ['ZO', 'D', 'ZO_SA', 'M']

def read_structure(method='ZO_SA', seed=None, dataset=None, budget=None):
    # path = "/root/ZO-DARTS_light_82/exps/NAS-Bench-201-algos/result/" + method + "_" + dataset[:-5].lower() + "+" + str(seed) + '/'
    path = "../exps/NAS-Bench-201-algos/Penalty15_ZO+/ZO_SA_" + dataset[:-5].lower() + "_+" + str(seed) + '_' + str(
        budget) + '/'
    print(path)
    file = open(glob.glob(path + "*.log")[0], "r")
    # file = open('./test.log', 'r')
    # 打开并读取每个文件的内容
    content = file.readlines()
    structure_recorder = ''
    for line in content:
        if method == 'ZO_SA':
            if '<<<--->>> The 039-050-th epoch : Structure' in line:
                structure_recorder = line
        else:
            if "last-geno" in line:
                structure_recorder = line
    print(structure_recorder)
    structures = structure_recorder.split('Structure(4 nodes with ')[1:]
    structures[0] = structures[0][:-2] # Delete excessive brackets
    return structures[0]

# print(read_structure(seed=1, dataset='OrganSMNIST', budget=100000))
#
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
    dataset = xargs.dataset
    cell_number = 3
    structure = read_structure(xargs.method, xargs.rand_seed, xargs.dataset, xargs.budget)

    print('batch_size:{:}, learning_rate:{:}, momentum:{:}, weight_decay:{:}, '
          'report_freq:{:}, mixup_alpha:{:},cutmix_alpha:{:}, num_classes:{:}, '
          'workers:{:}, dataset:{:}, lr_stepsize:{:}, lr_gamma:{:}, lr_warmup_epochs:{:}, '
          'lr_warmup_method:{:}, lr_warmup_decay:{:}, lr_min:{:}, epochs:{:}, label_smoothing:{:}, cell_number:{:}. structure:{:}'.format(
        batch_size, learning_rate, momentum, weight_decay, report_freq, mixup_alpha,
        cutmix_alpha, num_classes, workers, dataset, lr_stepsize, lr_gamma, lr_warmup_epochs,
        lr_warmup_method, lr_warmup_decay, lr_min, epochs, label_smoothing, cell_number, structure))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = None
    net = DiscreteNetworkSPARSEZOANNEAL(C=16, N=cell_number, max_nodes=4, num_classes=num_classes,
                                        search_space='nas-bench-201',
                                        structure=structure, affine=False, track_running_stats=False, params=params)
    net.to(device)

    # # 加载CIFAR-10数据集
    # train_data, test_data, xshape, class_num = get_datasets(
    #     "cifar10", "/root/autodl-fs/ZO_DARTS_light_test/nasbench201/dataset/cifar", -1
    # )
    datapath = "../nasbench201/dataset/" + xargs.dataset.lower() + ".npz"
    train_data, test_data, xshape, class_num = get_datasets(
        dataset, datapath, -1, mode='retrain_nopipe'
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
            raise RuntimeError(f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported.")
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers=[warmup_lr_scheduler, main_lr_scheduler],milestones=[lr_warmup_epochs])
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
    parser.add_argument("--method", type=str, default='ZO_SA', help="method")
    parser.add_argument("--budget", type=int, default=1000000, help='resource budget')
    args = parser.parse_args()
    main(args)