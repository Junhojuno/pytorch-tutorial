import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from torch.nn import functional as F

import pytorch_lightning as pl


class ImageClassifier(pl.LightningModule):
    pass


class ResNet(pl.LightningModule):
    """Not Fully implemented, just simple network"""
    
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = nn.Linear(in_features=28 * 28, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=64)
        self.layer3 = nn.Linear(in_features=64, out_features=10)
        self.dropout = nn.Dropout(p=0.1)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        output1 = F.relu(self.layer1(x))
        output2 = F.relu(self.layer2(output1))
        output3 = self.dropout(output1 + output2)
        logits = self.layer3(output3)
        return logits

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        batch_size = x.shape[0]
        # x = x.view(batch_size, -1).cuda() # (bs, 28*28) -> this line is for pytorch
        x = x.view(batch_size, -1) # (bs, 28*28) -> pytorch-lightning put it on the correct devices, don't worry about .cuda()
        
        logits = self(x)
        
        loss = self.loss_fn(logits, y) # pytorch_lightning does detach() automatically
        
        return loss
        
    def train_dataloader(self):
        dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
        # train_ds, val_ds = random_split(dataset, lengths=[int(round(len(dataset)*ratio, 0)), int(round(len(dataset)*(1.0 - ratio), 0))])
        train_loader = DataLoader(dataset=dataset, batch_size=128)
        # val_loader = DataLoader(dataset=val_ds, batch_size=batch_size)
        return train_loader