import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from model import iCaRLNet
from data_loader import iMNIST

def show_images(images):
    N = images.shape[0]
    fig = plt.figure(figsize=(1, N))
    gs = gridspec.GridSpec(1, N)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.show()

# Hyper Parameters
tasks = [[0, 1], [2, 3, 4], [5, 6], [7, 8], [9]]
num_classes = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

K = 100  # total number of exemplars
icarl = iCaRLNet(2048, len(tasks[0]))
icarl.cuda()

icarl.n_classes = 0

for task in tasks:
    icarl.n_classes += len(task)
    print(f"Loading training examples for classes {task}")
    train_set = iMNIST(root='./data', classes=task, train=True, download=True, transform=transform)
    test_set = iMNIST(root='./data', classes=task, train=False, download=True, transform=transform_test)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True)

    # Update representation via BackProp
    icarl.update_representation(train_set)
    m = K // icarl.n_classes

    # Reduce exemplar sets for known classes
    icarl.reduce_exemplar_sets(m)

    # Construct exemplar sets for new classes
    for y in range(icarl.n_known, icarl.n_classes):
        print(f"Constructing exemplar set for class-{y}")
        images = np.array([train_set[i][1].squeeze().numpy() for i in range(len(train_set)) if train_set[i][2] == y])
        icarl.construct_exemplar_set(images, m, transform)

    for y, P_y in enumerate(icarl.exemplar_sets):
        print(f"Exemplar set for class-{y}: {P_y.shape}")
        # show_images(P_y[:10])

    icarl.n_known = icarl.n_classes
    print(f"iCaRL classes: {icarl.n_known}")

    total = 0.0
    correct = 0.0
    for indices, images, labels in train_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels.cpu()).sum()

    print('Train Accuracy: %d %%' % (100 * correct / total))

    total = 0.0
    correct = 0.0
    for indices, images, labels in test_loader:
        images = Variable(images).cuda()
        print(images.shape)
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()

    print('Test Accuracy: %d %%' % (100 * correct / total))
