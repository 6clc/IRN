import voc12
import torch
from misc import  torchutils, imutils
from torch import multiprocessing
from config import *
from net.resnet50_cam import CamNet
from torch.utils.data import DataLoader
from torch import cuda
from torch.nn import functional as F
import numpy as np
import os
# import multiprocessing


def _work(process_id, model, dataset):
    print('idx', process_id)
    databin = dataset[process_id]
    data_loader = DataLoader(databin, shuffle=False, num_workers= num_workers, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack['name'][0] # 因为被pytorch重新封装了
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

    
            outputs = [ model.make_cam(img[0].cuda(non_blocking=True))
                        for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


if __name__ == '__main__':
    model = CamNet()
    model.load_state_dict(torch.load(cam_ckpt)['state_dict'], strict=True)
    # model.load_state_dict(torch.load(cam_ckpt), strict=True)

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(val_list,
                                                            voc12_root=voc12_root,
                                                            scales=cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)


    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset), join=True)
    print(']')

    torch.cuda.empty_cache()

