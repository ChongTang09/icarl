from torchvision.datasets import MNIST
import numpy as np
import torch
from PIL import Image

class iMNIST(MNIST):
    def __init__(self, root, classes=range(10), train=True, transform=None, target_transform=None, download=False):
        super(iMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.data = torch.stack(train_data)
            self.targets = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])

            self.data = torch.stack(test_data)
            self.targets = test_labels

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        return len(self.data)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """
        self.data = torch.cat((self.data, images), axis=0)
        self.targets = self.targets + labels
