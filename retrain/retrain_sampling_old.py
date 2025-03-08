import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from xautodl.utils import get_model_infos
from xautodl.models import get_search_spaces
from xautodl.models.cell_searchs.retrain_model import DiscreteNetworkSPARSEZOANNEAL
from xautodl.datasets import get_datasets
from cutmix_transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
from xautodl.config_utils import load_config
import os, sys, time, glob, argparse
import math

sys.path.append('..')
import StructureSampler

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

class_num_dict = {'OrganSMNIST': 11,
                  'OrganCMNIST': 11,
                  'OrganAMNIST': 11,
                  'OCTMNIST': 4,
                  'PneumoniaMNIST': 2,
                  'BreastMNIST': 2,
                  'BloodMNIST': 8,
                  'TissueMNIST': 8,
                  'DermaMNIST': 7,
                  'PathMNIST': 9}

performance_dict = {'OrganSMNIST': 0.55,
                    'OrganCMNIST': 0.7,
                    'OrganAMNIST': 0.7,
                    'OCTMNIST': 0.6,
                    'PneumoniaMNIST': 0.75,
                    'BreastMNIST': 0.65,
                    'BloodMNIST': 0.7,
                    'TissueMNIST': 0.4,
                    'DermaMNIST': 0.55,
                    'PathMNIST': 0.6}

# def getPath(_method, dataset, rand_seed):
#     method = _method.split('-')[0]
#     file_path = '../exps/NAS-Bench-201-algos/Others/{:}_{:}_+{:}/'.format(method, dataset[:-5].lower(), rand_seed)
#     return file_path

def getPath_ZOplus(dataset, rand_seed):
    file_path = '../exps/NAS-Bench-201-algos/ZO+/ZO_SA_{:}_+{:}/'.format(dataset[:-5].lower(), rand_seed)
    return file_path



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
    epochs = 300
    label_smoothing = 0.1


    # file_path = getPath_ZO(xargs.method, xargs.dataset, xargs.rand_seed)
    file_path = getPath_ZOplus(xargs.dataset, xargs.rand_seed)
    arch, _, _ = StructureSampler.read_prob(file_path)

    params = None
    search_space = get_search_spaces("cell", 'nas-bench-201-varied')

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

    counter = 0
    while counter < epochs:
        structure = StructureSampler.structure_generator_old(arch)

        print('batch_size:{:}, learning_rate:{:}, momentum:{:}, weight_decay:{:}, '
              'report_freq:{:}, mixup_alpha:{:},cutmix_alpha:{:}, num_classes:{:}, '
              'workers:{:}, dataset:{:}, lr_stepsize:{:}, lr_gamma:{:}, lr_warmup_epochs:{:}, '
              'lr_warmup_method:{:}, lr_warmup_decay:{:}, lr_min:{:}, epochs:{:}, label_smoothing:{:}, structure:{:}'.format(
            batch_size, learning_rate, momentum, weight_decay, report_freq, mixup_alpha,
            cutmix_alpha, num_classes, workers, xargs.dataset, lr_stepsize, lr_gamma, lr_warmup_epochs,
            lr_warmup_method, lr_warmup_decay, lr_min, epochs, label_smoothing, structure))

        net = DiscreteNetworkSPARSEZOANNEAL(C=16, N=3, max_nodes=4, num_classes=num_classes,
                                                 search_space=search_space,
                                                 structure=structure, affine=False, track_running_stats=False,
                                                 params=params)
        net.to(device)

        optimizer = optim.SGD(net.parameters(), learning_rate, momentum, weight_decay)
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

        recorder = 0
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
            counter += 1
            if epoch == 0:
                recorder = correct / total
            elif epoch == 19:
                if correct / total < 0.3:
                    print('The model learned nothing! Starting a new model.')
                    print('*' * 50)
                    counter = 0
                    recorder = 0
                    break
                elif math.fabs(correct / total - recorder < 1e-2) and correct / total < 0.8:  # Examine if the model didn't learn anything.
                    print('The model learned nothing! Starting a new model.')
                    print('*' * 50)
                    counter = 0
                    recorder = 0
                    break


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
    parser.add_argument("--method", type=str, default='DARTS', help='search algorithm')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)

