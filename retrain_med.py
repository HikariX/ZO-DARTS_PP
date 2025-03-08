import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from xautodl.models.cell_searchs.retrain_model import DiscreteNetworkSPARSEZOANNEAL
from xautodl.datasets import get_datasets

batch_size = 96
learning_rate = 0.025
momentum = 0.9
weight_decay = 3e-4
epochs = 200
report_freq = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# params = torch.load('SPARSEZOAnneal+2+0.8/checkpoint/seed-2-basic.pth')
params = None
structure = '|avg_pool_3x3~0|+|nor_conv_1x1~0|none~1|+|nor_conv_3x3~0|skip_connect~1|skip_connect~2|'
net = DiscreteNetworkSPARSEZOANNEAL(C=16, N=5, max_nodes=4, num_classes=10, search_space='nas-bench-201',
                                    structure=structure, affine=False, track_running_stats=False, params=params)
net.to(device)

# 加载CIFAR-10数据集
# train_data, test_data, xshape, class_num = get_datasets(
#        "cifar10", "/root/autodl-fs/ZO_DARTS_light_test/nasbench201/dataset/cifar", -1
#    )
train_data, test_data, xshape, class_num = get_datasets(
    "OrganAMNIST", "../nasbench201/dataset/organamnist.npz", -1
)

trainloader = torch.utils.data.DataLoader(train_data, batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(test_data, batch_size,
                                         shuffle=False, num_workers=2)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), learning_rate, momentum, weight_decay)

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

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))