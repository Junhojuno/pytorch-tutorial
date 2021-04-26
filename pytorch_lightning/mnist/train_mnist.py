"""Implement MNIST dataset training procedures simply."""
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# from dataset import load_mnist_dataset, train_val_split
from model import ResNet

import pytorch_lightning as pl


def main():
    model = ResNet()
    
    trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=5, gpus=1)
    trainer.fit(model)



if __name__ == '__main__':
    main()
