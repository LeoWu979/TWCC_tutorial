import logging
import os

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function


class vgg_cifar10_module(nn.Module):

    def __init__(self, num_classes=10):
        super(vgg_cifar10_module, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            
            nn.Linear(1024, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            
            nn.Linear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax(dim=1)
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            10: {'lr': 1e-3},
            20: {'lr': 5e-4},
            30: {'lr': 1e-4},
            40: {'lr': 5e-5},
            50: {'lr': 1e-5}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512*4*4)
        x = self.classifier(x)
        return x


def vgg_cifar10(**kwargs):
    args = kwargs.pop('args', None)

    num_classes = getattr(kwargs,'num_classes', 10)

    return vgg_cifar10_module(num_classes)
