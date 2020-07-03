

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from net.resnet50_cam import CamNet
from pytorch_lightning.loggers import TensorBoardLogger
from config import *


if  __name__ == '__main__':
    logger = TensorBoardLogger(log_dir + "/tb_logs", name="cam")

    model = CamNet()

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_dir+"/cam",
        save_top_k=True,
        verbose=True,
        monitor='fbeta2',
        mode='max',
        prefix=''
    )

    trainer = Trainer(gpus=1, checkpoint_callback=checkpoint_callback, max_epochs=5, logger=logger)
    trainer.fit(model)
