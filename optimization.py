# ******************************************************
# Author       : liuyang
# Last modified:	2020-01-15 15:54
# Email        : gxly1314@gmail.com
# Filename     :	new_test.py
# Description  : 
# ******************************************************
from __future__ import absolute_import
import sys
import argparse
import numpy as np
import torch
import scipy.io as sio
import datetime
import os
import cv2
import torch.backends.cudnn as cudnn
from core.workspace import register, create, global_config, load_config
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
#from modelling.ops.nms.nms_wrapper import nms
from utils.nms.nms_wrapper import nms
from data import anchor_utils
from tqdm import tqdm
from evaluation.evaluate_ap50 import evaluation_ap50


parser = argparse.ArgumentParser(description='Test Details')
parser.add_argument('--num_iter', '-n', default=140, type=int, help='number of iteration for test.')
parser.add_argument('--nms_th', default=0.3, type=float, help='nms threshold.')
parser.add_argument('--pre_nms_top_k', default=5000, type=int, help='number of max score image.')
parser.add_argument('--score_th', default=0.01, type=float, help='score threshold.')
parser.add_argument('--max_bbox_per_img', default=750, type=int, help='max number of det bbox.')
parser.add_argument('--config', '-c', default='./config.yml', type=str, help='config yml.')
parser.add_argument('--sub_project_name', default=None, type=str, help='sub_project_name.')
parser.add_argument('--backbone_cfg_file', '-bcf', default=None, type=str, help='backbone config file')
parser.add_argument('--test_idx',  default=None, type=int)



def detect_face(image, shrink):
    # starttime = datetime.datetime.now()
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)


    print('shrink:{}'.format(shrink))

    width = x.shape[1]
    height = x.shape[0]
    print('width: {}, height: {}'.format(width, height))

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = Variable(x.cuda(), volatile=True)

    torch.onnx.export(
        net, 
        x,
        "model.onnx",
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        export_params=True,
        verbose=True,
    )



def gen_dir(dir_name_list):
    for dir_name in dir_name_list:
        if not os.path.exists(dir_name):
            os.system('mkdir -p {}'.format(dir_name))

if __name__ == '__main__':
    args = parser.parse_args()
    # generate det_info and det_result
    cfg = load_config(args.config)
    cfg['phase'] = 'test'
    if 'use_hcam' in cfg and cfg['use_hcam']:
        # test_th
        cfg['fp_th'] = 0.12

    config_name = args.config.split('/')[-1].split('.')[-2]
    snapshots_dir = os.path.join('./snapshots', config_name)


    net = create(cfg.architecture)
    model_name = os.path.join(snapshots_dir, 'model_{}000.pth'.format(args.num_iter))
    print ('Load model from {}'.format(model_name))
    net.load_state_dict(torch.load(model_name))
    net.cuda()
    net.eval()
    print ('Finish load model.')

    # generate predict bbox
    img = np.zeros((1280, 720, 3)).astype(np.uint8)
    with torch.no_grad():
        max_im_shrink = (0x7fffffff / 200.0 / (img.shape[0] * img.shape[1])) ** 0.5 # the max size of input image for caffe
        max_im_shrink = 2.2 if max_im_shrink > 2.2 else max_im_shrink
        shrink = max_im_shrink if max_im_shrink < 1 else 1
        det0 = detect_face(img, shrink)  # origin test
