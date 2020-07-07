

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from net.resnet50_cam import CamNet
from pytorch_lightning.loggers import TensorBoardLogger
from config import *
import argparse


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-train",  default="0", type=int)
    args = parser.parse_args()
    epoch = args.train

    logger = TensorBoardLogger(log_dir + "/tb_logs", name="cam")
    model = CamNet()
    checkpoint_callback = ModelCheckpoint(
        filepath=f"{checkpoint_dir}/cam/ckpt",
        save_top_k=True,
        verbose=True,
        save_last=True,
        monitor='val_loss',
        mode='min',
        prefix='',
    )

    trainer = Trainer(default_root_dir=f"{checkpoint_dir}/cam/ckpt", 
                    gpus=n_gpus, 
                    checkpoint_callback=checkpoint_callback, 
                    max_epochs=epoch, 
                    logger=logger)
    trainer.fit(model)

    trainer = Trainer(gpus=1)
    trainer.test(model)
