import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from solution.data import load_datasets, INPUT_IMG_SIZE

MODEL_DEST_PATH = os.path.join(os.getcwd(), 'model')

EPSILON = 0.0000001

def initCNN(class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CNN(INPUT_IMG_SIZE, len(class_names)).to(device)
    return net, device


def saveMyModel(net):
    print('Saving the model serialization to', MODEL_DEST_PATH)

    torch.save(net.state_dict(), MODEL_DEST_PATH)

def loadMyModel(class_names):
    model, device = initCNN(class_names)
    model.load_state_dict(torch.load(MODEL_DEST_PATH))
    model.eval()
    return model, device


class BatchNorm(nn.Module):
    def __init__(self, channels):
        super(BatchNorm, self).__init__()
        self.weights = nn.Parameter(torch.zeros(channels).uniform_())
        self.biases = nn.Parameter(torch.zeros(channels, dtype=torch.float32))

    def forward(self, data):
        imgs = data
        weights = self.weights.view(-1)
        biases = self.biases.view(-1)

        def expanded(tens):
            tens = tens.unsqueeze(0) # clone for each batch member
            tens = tens.unsqueeze(-1) # clone for each pixel in row
            tens = tens.unsqueeze(-1) # clone for each pixel in column
            return tens.expand(imgs.shape)

        means = torch.mean(imgs, [0, 2, 3])
        diff = imgs - expanded(means)
        var2 = torch.sum(diff * diff, [0, 2, 3]) / (diff.numel() / len(diff))

        normal = diff / expanded(torch.sqrt(var2 + EPSILON))
        return ((normal * expanded(weights)) + expanded(biases)).view(data.shape)

class CNN(nn.Module):
    def __init__(self, imgSize, classCount):
        super(CNN, self).__init__()        

        # shape: (in_chann, out_chann, kernel_size, stride, padding, pooling)
        # IMPORTANT = in_chann for n+1th elem must equal out_chann of nth one
        CONV_DESCS = [
            (3, 18, 3, 1, 1, 2),
            (18, 30, 3, 1, 1, 2),
            (30, 50, 3, 1, 1, 2),
            (50, 80, 3, 1, 1, 2)
        ]

        HIDDEN_FC_SIZES = [6000]

        def imgSizeAfterConvAndPool(imgSize, kernel_size, padding, stride, pooling):
            return int((int((imgSize - kernel_size + 2*(padding)) / stride) + 1) / pooling)
        
        self.convs = nn.ModuleList([])
        for in_chann, out_chann, kernel_size, padding, stride, pooling in CONV_DESCS:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_chann, out_chann, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
                BatchNorm(out_chann),
                nn.MaxPool2d(kernel_size=pooling, stride=pooling)
            ))
            imgSize = imgSizeAfterConvAndPool(imgSize, kernel_size, padding, stride, pooling)
            self.convOutputSize = imgSize * imgSize * out_chann

        hiddenFcs = [self.convOutputSize] + HIDDEN_FC_SIZES + [classCount]

        self.fcs = nn.ModuleList([nn.Linear(s1, s2) for s1, s2 in zip(hiddenFcs[:-1], hiddenFcs[1:])])
    
    def forward(self, out):
        for conv in self.convs:
            out = conv(out)
        
        out = out.view(-1, self.convOutputSize)
        
        for fc in self.fcs[:-1]:
            out = F.relu(fc(out))

        return self.fcs[-1](out)
