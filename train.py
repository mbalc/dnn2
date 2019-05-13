import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from solution.model import saveMyModel, initCNN
from solution.data import load_datasets, INPUT_IMG_SIZE

EPOCH_COUNT = 2

TESTING_RATE = 1 # we execute testing sessions this many times during an epoch
LOGGING_PERIOD = 50 # there will be this many logs per train epoch

if TESTING_RATE < 1:
    raise Exception('Testing rate has to be positive')


executionStart = time.time()

image_datasets, dataloaders, dataset_sizes, class_names = load_datasets()
net, device = initCNN(class_names)

def train():
    net.train()
    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.94)

    batchCount = len(dataloaders['Training'])

    for batchId, (data, target) in enumerate(dataloaders['Training']):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        out = net(data)
        loss = crit(out, target)
        loss.backward()

        optimizer.step()

        if batchId % int(batchCount / LOGGING_PERIOD) == 0 or batchId + 1 == batchCount:
            print('({:.3f}s)\tTrain Epoch: {}\t[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                time.time() - executionStart, epoch + 1, batchId * len(data), len(image_datasets['Training']), # minor bug - last log for 100% has wrong image count
                100. * batchId / batchCount, loss.item())) # TODO reformat logging

        if batchId > 0 and batchId % int(batchCount / TESTING_RATE) == 0:
            test()

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
    test()
    
saveMyModel(net)