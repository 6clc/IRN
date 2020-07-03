from net.resnet50_irn import AffinityDisplacementLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os

if __name__ == '__main__':
    model = AffinityDisplacementLoss()

    trainer = Trainer(gpus=1, fast_dev_run=False, max_epochs=3)
    trainer.fit(model)
    trainer.save_checkpoint('irn.ckpt')