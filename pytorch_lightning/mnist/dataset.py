import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


def load_mnist_dataset():
    return datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())


def train_val_split(dataset, batch_size=32, ratio=0.7):
    train_ds, val_ds = random_split(dataset, lengths=[int(round(len(dataset)*ratio, 0)), int(round(len(dataset)*(1.0 - ratio), 0))])
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size)
    return train_loader, val_loader


if __name__ == '__main__':
    dataset = load_mnist_dataset()
    # ratio = 0.9
    # print([int(round(len(dataset)*ratio, 0)), int(round(len(dataset)*(1.0 - ratio), 0))])
    print(f'total number of data : {len(dataset)}')
    train_loader, val_loader = train_val_split(dataset, 32, 0.9)
    print(f'total number of train batches : {len(train_loader)}')
    print(f'total number of validation bathes : {len(val_loader)}')