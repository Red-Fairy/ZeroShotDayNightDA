#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os.path as osp

import cv2
import time
import glob
import numpy as np
from PIL import Image
import scipy.io as sio

from enhancement import EnhanceNet as PreProcessor
from torch.autograd import Variable

import torchvision.utils as vutils
import random

import tqdm
from draw_func import draw_hist, draw_hist_grad

pre_processor = PreProcessor(2).cuda()
pre_processor.eval()

result = "./Enh-Vis/"

net_param = torch.load("../../transfer/weights-enh-e4/vgg_rjig33(1.000000)/trainer_17000.pth", map_location=lambda storage, loc: storage)

new_net_param = {}
for param_key in net_param:
    if param_key[:6] == 'EnhNet':
        new_net_param[param_key[7:]] = net_param[param_key]
pre_processor.load_state_dict(new_net_param, strict=False)    

if not os.path.exists(result):
    os.mkdir(result)

process = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

def NormBack(image):
    mean = image.new_tensor([0.43216, 0.394666, 0.37645]).view(-1, 1, 1)
    std = image.new_tensor([0.22803, 0.22145, 0.216989]).view(-1, 1, 1)
    return image * std + mean

def to_bgr_chw(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # BGR to RGB
    image = image[[2, 1, 0], :, :]
    # CHW to HWC
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 0)
        image = np.swapaxes(image, 1, 2)
    return image

video_list = [f'/mnt/netdisk/wangwenjing/Datasets/ARID-images/ARID_split1_test/{i}_0.png' for i in ["Jump_8_12", "Wave_13_12", "Pour_1_17", "Pour_1_5", "Push_11_9", "Push_4_31", "Push_4_57", "Push_7_30", "Walk_6_15", "Walk_6_21", "Walk_6_37", "Walk_6_38", "Walk_6_39", "Walk_6_42"]]
# in_list.reverse()

for video_file in tqdm.tqdm(video_list):
    in_list = glob.glob(video_file.replace("_0.png", "_*.png"))

    frame_stack = []

    for in_file in in_list:
        image = Image.open(in_file)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = process(image).unsqueeze(0).unsqueeze(2).cuda()
        frame_stack.append(image)

    frame_stack = torch.cat(frame_stack, 2)
    frame_stack, curve = pre_processor(frame_stack, True)

    draw_hist(curve, "Enh-Vis/"+os.path.splitext(os.path.basename(in_file))[0]+"_curve.pdf")

    # image = pre_processor(image)

    # video = NormBack(frame_stack[:,:,-1,:,:]) * 255
    # frame = NormBack(image[:,:,0,:,:]) * 255

    # print(video-frame)
    # video = to_bgr_chw(video.data.cpu().numpy()[0, :, :, :].astype('uint8'))
    # frame = to_bgr_chw(frame.data.cpu().numpy()[0, :, :, :].astype('uint8'))
    # cv2.imwrite("frame.png", frame)
    # cv2.imwrite("video.png", video)
    # break

    for idx, in_file in enumerate(in_list):
        target = result+os.path.basename(in_file)
        if os.path.exists(target):
            continue
        image_enhanced = frame_stack[:,:,idx,:,:]
        image_enhanced = NormBack(image_enhanced)

        image_enhanced = image_enhanced.data.cpu().numpy()[0, :, :, :] * 255
        image_enhanced = image_enhanced.astype('uint8')
        image_enhanced = to_bgr_chw(image_enhanced)

        cv2.imwrite(target, image_enhanced)
