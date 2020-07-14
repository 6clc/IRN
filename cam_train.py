

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from net.resnet50_cam import CamNet
from pytorch_lightning.loggers import TensorBoardLogger
from config import *
import argparse
import torch


if  __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-train",  default="0", type=int)
    # args = parser.parse_args()
    # epoch = args.train
    epoch = cam_epoch

    logger = TensorBoardLogger(summary_writer_dir + "/cam_logs")

    checkpoint_callback = ModelCheckpoint(
        filepath=summary_writer_dir + "/cam_ckpt",
        save_top_k=True,
        verbose=True,
        save_last=True,
        monitor='val_loss',
        mode='min',
        prefix='',
    )

    trainer = Trainer(gpus=n_gpus, 
                    checkpoint_callback=checkpoint_callback, 
                    max_epochs=epoch, 
                    logger=logger)


    model = CamNet()                
    trainer.fit(model)

    trainer = Trainer(gpus=1)
    if cam_ckpt is not None:
        model.load_state_dict(torch.load(cam_ckpt)['state_dict'], strict=True)
    trainer.test(model)
