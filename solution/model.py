import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from solution.data import load_datasets, INPUT_IMG_SIZE

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
np.random.seed(42)
random.seed(42)

MODEL_DEST_PATH = os.path.join(os.getcwd(), 'model')

DEPTH = 3

RGB_CHANNEL_COUNT = 3
CLASS_COUNT = 30

EPSILON = 0.0000001

def initFCN():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FCN().to(device)
    return net, device

def saveMyModel(net):
    print('Saving the model serialization to', MODEL_DEST_PATH)

    torch.save(net.state_dict(), MODEL_DEST_PATH)

def loadMyModel(class_names):
    model, device = initCNN(class_names)
    model.load_state_dict(torch.load(MODEL_DEST_PATH))
    model.eval()
    return model, device

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        CHANNELS = [None, 64, 128, 256, 512]
        KERNEL_SIZE = 3
        PADDING = (KERNEL_SIZE - 1) // 2
        
        leftconvs = []
        maxpools = []
        upscalers = []
        rightconvs = []
        for in_chann, out_chann in zip(CHANNELS[:-1], CHANNELS[1:]):
            if (in_chann) == None: in_chann = RGB_CHANNEL_COUNT
            leftconvs.append(nn.Sequential(
                nn.Conv2d(in_chann, out_chann, KERNEL_SIZE, padding=PADDING),
                nn.ReLU(),
                nn.Conv2d(out_chann, out_chann, KERNEL_SIZE, padding=PADDING),
                nn.ReLU(),
            ))

            maxpools.append(nn.MaxPool2d(2, stride = 2)) # 4 pixels -> 1 pixel

            upscalers.append(nn.ConvTranspose2d(2 * out_chann, out_chann, KERNEL_SIZE + 1, padding=PADDING, stride=2)) # TODO validate if this makes sense, parameterize

            if (in_chann) == RGB_CHANNEL_COUNT: in_chann = CLASS_COUNT
            rightconvs.append(nn.Sequential(
                nn.Conv2d(2 * out_chann, out_chann, KERNEL_SIZE, padding=PADDING), # channels from left part of U-net + from previous layer
                nn.ReLU(),
                nn.Conv2d(out_chann, out_chann, KERNEL_SIZE, padding=PADDING),
                nn.ReLU(),
            ))

        intermed_chann_count = CHANNELS[-1]

        self.leftconvs = nn.ModuleList(leftconvs)
        self.maxpools = nn.ModuleList(maxpools)

        self.intermediate = nn.Sequential(
            nn.Conv2d(intermed_chann_count, 2 * intermed_chann_count, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(2 * intermed_chann_count, 2 * intermed_chann_count, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
        )

        upscalers.reverse()
        self.upscalers = nn.ModuleList(upscalers)
        rightconvs.reverse()
        self.rightconvs = nn.ModuleList(rightconvs)
    
    def forward(self, imgs): # imgs.shape = (BATCH_SIZE, CHANNEL_COUNT, HEIGHT, WIDTH)
        output_buffer = [] # storage for data to crop up and provide to the right side of unet

        for (conv, pool) in zip(self.leftconvs, self.maxpools):
            imgs = conv(imgs)
            output_buffer.append(imgs)
            imgs = pool(imgs)

        imgs = self.intermediate(imgs)

        for (upscale, conv) in zip(self.upscalers, self.rightconvs):
            imgs = upscale(imgs)
            pre_downscale = output_buffer.pop() # TODO potential cropping
            imgs = torch.cat([imgs, pre_downscale], dim=1) # merge channel-wise
            imgs = conv(imgs)

        return imgs
