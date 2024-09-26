# ResNet-18 training script optimized for time-to-96%.

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from airbench import CifarLoader, evaluate

hyp = {
    'opt': {
        'epochs': 150,
        'batch_size': 512,
        'lr': 0.2,
        'momentum': 0.9,
        'wd': 1e-3,
    },
    'aug': {
        'flip': True,
        'translate': 4,
        'cutout': 0,
    },
}

#############################################
#            Network Components             #
#############################################

'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        widths = [64, 128, 256, 512]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(widths[3], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def make_rn18():
    model = ResNet18()
    model = model.cuda().to(memory_format=torch.channels_last)
    for m in model.modules():
        if type(m) is not nn.BatchNorm2d:
            m.half()
    return model

########################################
#              Training                #
########################################

def train(train_loader, test_loader=None, epochs=hyp['opt']['epochs'], lr=hyp['opt']['lr']):

    if test_loader is None:
        test_loader = CifarLoader('cifar10', train=False, batch_size=1000)
    batch_size = train_loader.batch_size

    momentum = hyp['opt']['momentum']
    wd = hyp['opt']['wd']

    total_train_steps = len(train_loader) * epochs
    lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.1 * total_train_steps), total_train_steps],
                            [0.2, 1, 0]) # Triangular learning rate schedule

    model = make_rn18()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr/batch_size, momentum=momentum, nesterov=True,
                                weight_decay=wd*batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    train_loss, train_acc, test_acc = [], [], [torch.nan]

    it = tqdm(range(epochs))
    for epoch in it:

        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            train_loss.append(loss.mean().item())
            train_acc.append((outputs.detach().argmax(1) == labels).float().mean().item())
            optimizer.zero_grad(set_to_none=True)
            loss.sum().backward()
            optimizer.step()
            scheduler.step()
            it.set_description('Training loss=%.4f acc=%.4f' % (train_loss[-1], train_acc[-1]))

    test_acc.append(evaluate(model, test_loader, tta_level=0))
    print('tta_level=0 acc: %.4f' % test_acc[-1])
    test_acc.append(evaluate(model, test_loader, tta_level=1))
    print('tta_level=1 acc: %.4f' % test_acc[-1])
    test_acc.append(evaluate(model, test_loader, tta_level=2))
    print('tta_level=2 acc: %.4f' % test_acc[-1])

    return model

if __name__ == '__main__':

    train_loader = CifarLoader('cifar10', train=True, batch_size=hyp['opt']['batch_size'], aug=hyp['aug'])
    os.makedirs('models', exist_ok=True)
    for _ in range(10):
        model = train(train_loader)
        torch.save(model.state_dict(), 'models/%s.pt' % uuid.uuid4())

