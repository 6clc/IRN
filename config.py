import os

dataname = 'adas'
if dataname == 'adas':
    from configs.adasconfig import *
elif dataname == 'voc':
    from configs.vocconfig import *
elif dataname == 'camvid':
    from configs.camvidconfig import *

device_ids = "0, 1"
n_gpus = len(device_ids.split(','))
num_workers = 4*n_gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_ids

summary_writer_dir = f'/workspace/SummaryWriter/{dataname}'
score_writer_dir = f'/workspace/ScoreWriter/{dataname}'
train_list = voc12_root + '/train.txt'
val_list = voc12_root + '/val.txt'
test_list = voc12_root + '/val.txt'
cls_labels = voc12_root + "/onehot.npy"

assert n_classes == len(cam_w)


cam_epoch = 5
cam_ckpt = f'/workspace/SummaryWriter/{dataname}/last.ckpt'
cam_lr = 1e-4
cam_wd = 1e-4
cam_scales = (1.0, 0.5, 1.5, 2.0)
cam_bs = 32 #(512*512 bs=64 满载)

cam_out_dir = score_writer_dir + '/cam_seg_npys'
cam_seg_dir = score_writer_dir + '/cam_seg_pngs'

cam_eval_thres = 0.15  # 小于这个阈值的都被当做是背景
chainer_eval_set = 'val'

# conf_fg_thres = 0.30
# conf_bg_thres = 0.05
# ir_label_out_dir = result_dir + '/ir_label'

# irn_crop_size = 512
# irn_batch_size = 8


if not os.path.exists(cam_out_dir):
    os.makedirs(cam_out_dir)

if not os.path.exists(cam_seg_dir):
    os.makedirs(cam_seg_dir)

# if not os.path.exists(ir_label_out_dir):
#     os.makedirs(ir_label_out_dir)
