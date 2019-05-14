import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from solution.model import saveMyModel, initFCN
from solution.data import load_datasets, INPUT_IMG_SIZE

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
np.random.seed(42)
random.seed(42)

EPOCH_COUNT = 2

TESTING_RATE = 1 # we execute testing sessions this many times during an epoch
LOGGING_PERIOD = 50 # there will be this many logs per train epoch

if TESTING_RATE < 1:
    raise Exception('Testing rate has to be positive')


executionStart = time.time()

train_sampler, valid_sampler, train_loader, valid_loader = load_datasets()
net, device = initFCN()

def score(output, target):
    out_render = torch.argmax(output, dim=1)

    grad = out_render - target
    points = torch.nonzero(grad).size(0)
    max_points = grad.flatten().size(0)

    return 100 * (max_points - points) / max_points

def train():
    net.train()
    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    batch_count = len(train_loader)

    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        out = net(data)
        loss = crit(out, target)
        loss.backward()

        optimizer.step()

        print('(E{} {:.3f}s)\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tScore:  {:.3f}'.format(
            epoch + 1, time.time() - executionStart, batch_id * len(data), len(train_sampler), # minor bug - last log for 100% has wrong image count
            100. * batch_id / batch_count, loss.item(), score(out, target)))


def test():
    net.eval()
    test_loss = 0
    correct = 0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in dataloaders['Test']:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += crit(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(image_datasets['Test'])

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(image_datasets['Test']),
        100. * correct / len(image_datasets['Test'])))
            

for epoch in range(EPOCH_COUNT):
    train()
    
saveMyModel(net)
