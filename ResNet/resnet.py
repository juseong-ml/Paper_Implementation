import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from . import block

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        #inputs = 64x224x224
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #outputs.shape = 64x112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #inputs = 64x112x112
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #outputs = 64x56x56
        self.layer1 = self._make_layer(block, 64, layer[0])
        self.layer2 = self._make_layer(block, 128, layer[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight,0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        #self._make_layer(bottleneck, 64,3)
        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride), #conv1x1(64,256,1)
                    nn.BatchNorm2d(planes * block.expansion) #batchnorm2d(256)
                ) #downsample -> channel을 맞추기 위해
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
            #for _ in range(1, 3):
                layers.append(block(self.inplanes, planes)) #2번

            return nn.Sequential(*layers)
        #self.layer1 = [
           # Bottleneck(64,64,1,downsample)
            # Bottleneck(256,64)
            #Bottlenect(256,64) ]

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2,2,2,2], **kwargs) # 2*(2+2+2+2) + 1(conv1) + 1(fc) = 16+ 2
    return model

def resnet50(pretrained=False, **kwargs):
    model = ResNet(BottleNeck, [3,4,6,3], **kwargs) # 3 * (3+4+6+3) + 1(conv1) + 1(fc) = 50
    return model

def resnet152(pretrained=False, **kwargs):
    model ResNet(BottleNeck, [3,8,36,3], **kwargs) # 3 * (3+8+36+3) + 2 = 152