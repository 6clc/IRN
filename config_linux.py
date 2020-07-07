import os
device_ids = "0, 1"
n_gpus = len(device_ids.split(','))
num_workers = 4*n_gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_ids

dataname = 'voc'


log_dir = '/workspace/SummaryWriter'
checkpoint_dir = log_dir

result_dir = '/workspace/ScoreWriter'
# voc12_root = '/workspace/dataset/adas_parsing_data_0409_to0416/voc_fmt'
voc12_root = '/workspace/dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'


train_list = voc12_root + '/train.txt'
val_list = voc12_root + '/val.txt'
test_list = voc12_root + '/val.txt'
cls_labels = voc12_root + "/onehot.npy"
n_classes = 20

# cam_w = [0.1, 0.5, 0.1, 1, 0.4, 0.2]
# cam_w = [80, 15, 80, 8, 20, 40] 
cam_w = [1]*n_classes

assert n_classes == len(cam_w)
cam_ckpt = '/workspace/Projects/irn-master/sess/res50_cam.pth.pth'
cam_scales = (1.0, 0.5, 1.5, 2.0)
cam_bs = 32
cam_out_dir = result_dir+ '/cam'
cam_out_dir = '/workspace/Projects/irn-master/result/cam'

cam_eval_thres = 0.15
chainer_eval_set = 'val'

conf_fg_thres = 0.30
conf_bg_thres = 0.05
ir_label_out_dir = result_dir + '/ir_label'

irn_crop_size = 512
irn_batch_size = 8


if not os.path.exists(cam_out_dir):
    os.makedirs(cam_out_dir)

if not os.path.exists(ir_label_out_dir):
    os.makedirs(ir_label_out_dir)
