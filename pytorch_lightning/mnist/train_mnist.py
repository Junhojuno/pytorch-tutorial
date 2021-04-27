import torch
from model import ResNet

import pytorch_lightning as pl


def main():
    model = ResNet()
    
    trainer = pl.Trainer(progress_bar_refresh_rate=20, max_epochs=5, gpus=1)
    trainer.fit(model)



if __name__ == '__main__':
    main()
