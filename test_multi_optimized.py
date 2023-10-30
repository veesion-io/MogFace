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
from utils.nms.nms_wrapper import nms
from data import anchor_utils
from tqdm import tqdm
from evaluation.evaluate_ap50 import evaluation_ap50
from data.preprocess import BasePreprocess
from data.data_aug_settings import DataAugSettings
from tqdm import tqdm
from data.anchors_opr import GeneartePriorBoxes
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
trt.init_libnvinfer_plugins(None, "")
# Load TRT model
TRT_LOGGER = trt.Logger()
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#
import argparse
import os

import tensorrt as trt
import pycuda.driver as cuda
TRT_LOGGER = trt.Logger()

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

parser = argparse.ArgumentParser(description='Test Details')
parser.add_argument('--num_iter', '-n', default=140,
                    type=int, help='number of iteration for test.')
parser.add_argument('--nms_th', default=0.3, type=float, help='nms threshold.')
parser.add_argument('--pre_nms_top_k', default=5000,
                    type=int, help='number of max score image.')
parser.add_argument('--score_th', default=0.08,
                    type=float, help='score threshold.')
parser.add_argument('--max_bbox_per_img', default=750,
                    type=int, help='max number of det bbox.')
parser.add_argument('--scale_weight', default=15, type=float,
                    help='to differentiate the gap between large and small scale..')
parser.add_argument('--max_img_shrink', default=2.6,
                    type=float, help='constrain the max shrink of img.')
parser.add_argument('--vote_th', default=0.6, type=float,
                    help='bbox vote threshold')
parser.add_argument('--config', '-c', default='./config.yml',
                    type=str, help='config yml.')
parser.add_argument('--sub_project_name', default=None,
                    type=str, help='sub_project_name.')
parser.add_argument('--test_min_scale', default=0, type=int,
                    help='the min scale of det bbox')
parser.add_argument('--flip_ratio', default=None, type=float)
parser.add_argument('--test_hard', default=0, type=int)
parser.add_argument('--videos_path', default="/workspace/videos/", type=str)



def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def forward_pass(x, inputs, outputs, bindings, stream, context):
    x = x.transpose(2, 0, 1)[np.newaxis, :, :, :]
    inputs[0].host = np.ascontiguousarray(x, dtype=np.float32)
    [
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
        for inp in inputs
    ]

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def detect_face(
        image, shrink, inputs, outputs, bindings, stream,
                        context, anchors_function):
    # starttime = datetime.datetime.now()
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink,
                       interpolation=cv2.INTER_LINEAR)
    x = cv2.resize(image, (1280, 720),
                    interpolation=cv2.INTER_LINEAR)

    # print('shrink:{}'.format(shrink))

    width = x.shape[1]
    height = x.shape[0]
    # print('width: {}, height: {}'.format(width, height))

    out = forward_pass(x, inputs, outputs, bindings, stream, context)

    anchors = anchor_utils.transform_anchor((anchors_function(height, width)))
    anchors = torch.FloatTensor(anchors).cuda()
    decode_bbox = anchor_utils.decode(torch.from_numpy(out[1].reshape(-1, 4)).cuda(), anchors)
    boxes = decode_bbox
    scores = torch.from_numpy(out[0].reshape(-1, 1)).cuda()

    top_k = args.pre_nms_top_k
    v, idx = scores[:, 0].sort(0)
    idx = idx[-top_k:]
    boxes = boxes[idx]
    scores = scores[idx]

    # [11620, 4]
    boxes = boxes.cpu().numpy()
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    boxes[:, 0] /= shrink
    boxes[:, 1] /= shrink
    boxes[:, 2] = boxes[:, 0] + w / shrink - 1
    boxes[:, 3] = boxes[:, 1] + h / shrink - 1
    #boxes = boxes / shrink
    # [11620, 2]
    if int(args.test_min_scale) != 0:
        boxes_area = (boxes[:, 3] - boxes[:, 1] + 1) * \
            (boxes[:, 2] - boxes[:, 0] + 1) / (shrink * shrink)
        boxes = boxes[boxes_area > int(args.test_min_scale)**2]
        scores = scores[boxes_area > int(args.test_min_scale)**2]

    scores = scores.cpu().numpy()

    inds = np.where(scores[:, 0] > args.score_th)[0]
    if len(inds) == 0:
        det = np.empty([0, 5], dtype=np.float32)
        return det
    c_bboxes = boxes[inds]
    # [5,]
    c_scores = scores[inds, 0]
    # [5, 5]
    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
        np.float32, copy=False)

    #starttime = datetime.datetime.now()
    keep = nms(c_dets, args.nms_th)
    #endtime = datetime.datetime.now()
    #print('nms forward time = ',(endtime - starttime).seconds+(endtime - starttime).microseconds/1000000.0,' s')
    c_dets = c_dets[keep, :]

    max_bbox_per_img = args.max_bbox_per_img
    if max_bbox_per_img > 0:
        image_scores = c_dets[:, -1]
        if len(image_scores) > max_bbox_per_img:
            image_thresh = np.sort(image_scores)[-max_bbox_per_img]
            keep = np.where(c_dets[:, -1] >= image_thresh)[0]
            c_dets = c_dets[keep, :]
    return c_dets



def flip_test(image, shrink, inputs, outputs, bindings, stream,
                        context):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink,
        inputs, outputs, bindings, stream, context, anchors_function=anchors_function)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2] - 1
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0] - 1
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    det[:, :4] = np.round(det[:, :4])
    dets = []
    while det.shape[0] > 0:
        # IOU
        box_w = np.maximum(det[:, 2] - det[:, 0], 0)
        box_h = np.maximum(det[:, 3] - det[:, 1], 0)
        area = box_w * box_h
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = area[0] + area[:] - inter
        union[union <= 0] = 1
        o = inter / union
        o[0] = 1

        # get needed merge det and delete these det
        merge_index = np.where(o >= args.vote_th)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    if not len(dets):
        return dets
    if dets.shape[0] > 750:
        dets = dets[0:750, :]
    return dets


def write_to_txt(f, det, height, width, image_id):
    if not len(det):
        return
    for i in range(det.shape[0]):
        if det[i][0] < 0.0:
            xmin = 0.0
        else:
            xmin = det[i][0]

        if det[i][1] < 0.0:
            ymin = 0.0
        else:
            ymin = det[i][1]

        if det[i][2] > width - 1:
            xmax = width - 1
        else:
            xmax = det[i][2]

        if det[i][3] > height - 1:
            ymax = height - 1
        else:
            ymax = det[i][3]

        score = det[i][4]
        f.write('{} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(image_id, round(xmin), round(ymin), round(xmax - xmin + 1), round(ymax - ymin + 1), score))


def gen_soft_link_dir(dir_name_list):
    for dir_name in dir_name_list:
        cur_dir_name = dir_name.split('/')[-1]
        if os.path.exists(cur_dir_name):
            os.system('rm -rf ./{}'.format(cur_dir_name))
        if not os.path.exists(dir_name):
            raise ValueError(
                'Cannot create soft link, {} does not exist'.format(dir_name))
        os.system('ln -s {} ./{}'.format(dir_name, cur_dir_name))


def gen_dir(dir_name_list):
    for dir_name in dir_name_list:
        if not os.path.exists(dir_name):
            os.system('mkdir -p {}'.format(dir_name))


if __name__ == '__main__':
    args = parser.parse_args()
    args.max_img_shrink = 1.1
    args.vote_th = 0.5
    args.nms_th = 0.4
    args.scale_weight = 10
    args.flip_ratio = 1.4

    # generate det_info and det_result
    cfg = load_config(args.config)
    cfg['phase'] = 'test'

    config_name = args.config.split('/')[-1].split('.')[-2]
    snapshots_dir = os.path.join('./snapshots', config_name)

    det_info_dir = os.path.join('./det_info', config_name)

    det_result_dir = os.path.join('./det_result', config_name)

    save_info_dir_name = 'ss_' + str(args.num_iter) + '_nmsth_' + str(args.nms_th) + \
        '_scoreth_' + str(args.score_th)

    abs_save_dir = os.path.join(det_info_dir, save_info_dir_name)
    det_result_txt = os.path.join(det_result_dir, 'result.txt')

    gen_dir_list = [abs_save_dir, det_result_dir]
    gen_dir(gen_dir_list)

    with open("model.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    inputs, outputs, bindings, stream = allocate_buffers(engine)



    anchors_function = GeneartePriorBoxes(
        scale_list=[0.68], aspect_ratio_list=[1.0], stride_list=[4, 8, 16, 32, 64, 128], anchor_size_list=[16, 32, 64, 128, 256, 512])

    # generate predict bbox
    preprocess_fn = BasePreprocess(
        data_aug_settings=DataAugSettings(), normalize_pixel=True, use_rgb=True, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225])
    videos_path = args.videos_path
    output_directory = "annotations"
    videos = os.listdir(videos_path)
    # Forward pass
    with engine.create_execution_context() as context:
        for video_name in tqdm(videos):
            video_reader = cv2.VideoCapture(os.path.join(videos_path, video_name))
            is_frame_valid, img = video_reader.read()
            annotations_file_name = ".".join(video_name.split(".")[:-1])+".txt"
            annotations_file = os.path.join(
                output_directory, annotations_file_name)
            image_id = 0
            while is_frame_valid:
                img = preprocess_fn(img, phase='validation')
                with torch.no_grad():
                    # the max size of input image for caffe
                    max_im_shrink = (0x7fffffff / 200.0 /
                                    (img.shape[0] * img.shape[1])) ** 0.5
                    max_im_shrink = args.max_img_shrink if max_im_shrink > args.max_img_shrink else max_im_shrink
                    shrink = max_im_shrink if max_im_shrink < 1 else 1
                    det0 = detect_face(
                        img, shrink, 
                        inputs, outputs, bindings, stream,
                        context, anchors_function=anchors_function)  # origin test
                    det1 = flip_test(img, shrink, inputs, outputs, bindings, stream,
                        context)    # flip test

                # if args.flip_ratio is not None:
                #     det = np.row_stack((det0, det1, det2, det3, det4, det5))
                # else:
                det = np.row_stack((det0, det1))
                dets = bbox_vote(det)
                mode = "a" if os.path.exists(annotations_file) else "w"

                with open(annotations_file, mode) as f:
                    write_to_txt(f, dets, img.shape[0], img.shape[1], image_id)
                is_frame_valid, img = video_reader.read()
                image_id += 1
