from config import *
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from PIL import Image
import cv2 
rgb_dict = [[0, 0, 0]] + [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [ 255, 0, 255], [0, 255,255]]*20
# rgb_dict = [[0, 0, 0]] + [[0, 0, 0], [0, 255, 0], [0, 0, 0], [255, 255, 0], [ 255, 0, 255], [0, 255,255]]*20



if __name__ == "__main__":
    dataset = VOCSemanticSegmentationDataset(split=chainer_eval_set, data_dir=voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]


    preds = []
    for idx, id in enumerate(dataset.ids):
        cam_dict = np.load(os.path.join(cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams[:2,:]=0
        cams[3:, :]=0
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cam_eval_thres) # 添加背景的阈值
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant') # 添加背景
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
        # print(np.unique(preds[-1]), np.unique(labels[idx]))
        seg = np.zeros((preds[-1].shape[0], preds[-1].shape[1], 3), np.uint8)
        for i in range(n_classes):
            seg[preds[-1] == i] = rgb_dict[i]

        if dataname == 'camvid':
            ori_img = cv2.imread(voc12_root + '/JPEGImages/' + id + '.png')
        else:
            ori_img = cv2.imread(voc12_root + '/JPEGImages/' + id + '.jpg')
        seg = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)
      
        dst_img = cv2.addWeighted(ori_img, 0.2, seg, 0.8, 0)
        cv2.imwrite(cam_seg_dir + '/' + id + '.png', dst_img)


    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print({'iou': iou, 'miou': np.nanmean(iou)})
