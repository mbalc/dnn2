import os
import random
import time
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision

import numpy as np

from solution.model import loadMyModel, saveMyModel, initFCN
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
writer = SummaryWriter()


print("({:.3f}s) Preparing data...".format(time.time() - executionStart))
train_size, valid_size, train_loader, valid_loaders = load_datasets()
print("({:.3f}s) Preparing model...".format(time.time() - executionStart))
net, device = initFCN()
# net, device = loadMyModel()

print("({:.3f}s) Beginning training session...".format(time.time() - executionStart))

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


        iteration = (batch_count * epoch) + batch_id
        iter_score = score(out, target)

        writer.add_scalar('train_loss', loss.item(), iteration)
        writer.add_scalar('train_score', iter_score, iteration)
        print('(E{} {:.3f}s T)\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tScore:  {:.3f}%'.format(
            epoch + 1, time.time() - executionStart, batch_id * len(data), train_size, # minor bug - last log for 100% has wrong image count
            100. * batch_id / batch_count, loss.item(), iter_score))

def test():
    net.eval()
    losses = []
    scores = []
    correct = 0
    crit = nn.CrossEntropyLoss()
    batch_count = len(valid_loaders[0]) * len(valid_loaders)
    with torch.no_grad():
        for loader_id, dataloader in enumerate(valid_loaders):
            for batch_id, (data, target) in enumerate(dataloader):
                batch_id += loader_id * len(valid_loaders[0])
                data, target = data.to(device), target.to(device)
                output = net(data)
                l = crit(output, target).item()
                s = score(output, target)
                losses.append(l)
                scores.append(s)

                iteration = epoch * batch_count + batch_id
                writer.add_scalar('test_loss', l, iteration)
                writer.add_scalar('test_score', s, iteration)

                print('(E{} {:.3f}s V)\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tScore:  {:.3f}%'.format(
                    epoch + 1, time.time() - executionStart, batch_id * len(data), valid_size, # minor bug - last log for 100% has wrong image count
                    100. * batch_id / batch_count, l, s))

    avg_score = statistics.mean(scores)
    avg_loss = statistics.mean(losses)

    writer.add_scalar('test_total_loss', avg_loss, epoch)
    writer.add_scalar('test_total_score', avg_score, epoch)

    print('\nTest set: Average loss: {:.4f}, Score: ({:.2f}%)\n'.format(
        avg_loss, avg_score))
            

for epoch in range(EPOCH_COUNT):
    train()
    test()
    saveMyModel(net, '-epoch-' + str(epoch))

saveMyModel(net, '')

writer.close()
