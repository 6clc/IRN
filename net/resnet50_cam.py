import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
from pytorch_lightning.core.lightning import LightningModule
from torchvision.datasets import MNIST
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from voc12.dataloader import VOC12ClassificationDataset
from torchvision.models._utils  import IntermediateLayerGetter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import *
from metrics import *


class CamNet(LightningModule):

    def __init__(self):
        super(CamNet, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

        self.params = []
        self.freeze()
    
    def freeze(self):
        for name, para in self.named_parameters():
            # if 'conv' in name and 'resnet' in name: 
            #     para.requires_grad = False
            # else:
            self.params.append(para)

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, n_classes)

        return x
    
    def training_step(self, batch, batch_idx):
        img = batch['img']
        label = batch['label']
        y_hat = self(img)
        loss = F.multilabel_soft_margin_loss(y_hat, label)
        logs = {'train_loss':loss}

        return {'loss':loss, 'log':logs}
    
    def validation_step(self, batch, batch_idx):
        img = batch['img']
        label = batch['label']

        y_hat = self(img)
        loss = F.multilabel_soft_margin_loss(y_hat, label)
        tp, total_pred, total_targ = multilabel_fbeta_batch(y_hat, label)
        return {'val_loss': loss, 'tp':tp, 'total_pred':total_pred, 'total_targ':total_targ}
    
    def validation_epoch_end(self, outputs):
        # print(outputs, type(outputs))
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        fbeta = multilabel_fbeta_epoch(
            tp=torch.stack([x['tp'] for x in outputs]).sum(),
            pred=torch.stack([x['total_pred'] for x in outputs]).sum(),
            targ=torch.stack([x['total_targ'] for x in outputs]).sum()
        )
        logs = {'fbeta2': fbeta, 'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'fbeta2': fbeta, 'log':logs}
    
    def configure_optimizers(self):
        optimer = Adam([{'params':self.params}], lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimer, T_max=32)

        return  [optimer], [scheduler]
    
    def train_dataloader(self):
        train_dataset = VOC12ClassificationDataset(train_list, 
                                                    voc12_root=voc12_root,
                                                    resize_long=(320, 640), hor_flip=True,
                                                    crop_size=512, crop_method="random")
        loader = DataLoader(train_dataset, batch_size=16,
                                   shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)                                    
        return loader
    
    def val_dataloader(self):
        val_dataset = VOC12ClassificationDataset(val_list, voc12_root=voc12_root,
                                                              crop_size=512)
        loader = DataLoader(val_dataset, batch_size=16,
                                 shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

        return loader

    def make_cam(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1) # flip tricks
        return x

