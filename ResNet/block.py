import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import time
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

#conv3x3 / conv1x1 func
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,bias=False)

#BasicBlock Module : 2개의 3x3 conv layer와 skip connection으로 구성된 basic block module
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x) #3x3 / stride=stride
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 / stride =1

        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


#bottleneck module : 50개 이상의 layer를 가진 ResNet에서 계산 효율을 증가시키기 위해 3x3 conv layer 앞뒤로 1x1 conv layer를 추가한 bottleneck module
class BottleNeck(nn.Module):
    expansion = 4 # kernel size 키우기 위해

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self. conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes,planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) # 1x1 / stride=1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 / stride=stride
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1 / stride=1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample

        out += identity
        out = self.relu(out)

        return out
    


