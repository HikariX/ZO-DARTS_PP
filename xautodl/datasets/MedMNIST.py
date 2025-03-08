import os, sys, hashlib, torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt

class medMNIST(data.Dataset):
    def __init__(self, root, train, transform, use_num_of_class_only=None):
        self.root = root
        self.transform = transform
        self.train = train

        data = np.load(self.root)
        if self.train:
            train_images = data['train_images']
            val_images = data['val_images']
            self.data = np.concatenate((train_images, val_images), axis=0)
            train_labels = data['train_labels']
            val_labels = data['val_labels']
            targets = np.concatenate((train_labels, val_labels), axis=0).astype(np.uint8)
            self.targets = targets.reshape(targets.shape[0])
        else:
            self.data = data['test_images']
            targets = data['test_labels'].astype(np.uint8)
            self.targets = targets.reshape(targets.shape[0])

    def __repr__(self):
        return "{name}({num} images, {classes} classes)".format(
            name=self.__class__.__name__,
            num=len(self.data),
            classes=len(set(self.targets)),
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train = medMNIST("/Users/LightningX/.medmnist/organsmnist.npz", True , None)
    valid = medMNIST("/Users/LightningX/.medmnist/organsmnist.npz", False, None)
    data = np.load('/Users/LightningX/.medmnist/organsmnist.npz')
    print(data['train_images'].shape)
    print(data['val_images'].shape)
    img = data['val_images'][0]
    # plt.imshow(img)
    # plt.show()

    # (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # print(x_train.shape, type(train), data['train_images'].shape, data['val_images'].shape)
    # print(y_train.shape, data['train_labels'].shape)
    # image, label = train[111]
    # print(image.size)