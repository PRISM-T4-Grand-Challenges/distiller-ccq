##AlexNet fot CIFAR-10
##Taken from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/alexnet.py
##Planning to also try this https://github.com/AbhishekTaur/AlexNet-CIFAR-10/blob/master/alexnet.py
##Also check https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py
##For MobileNet https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py

import torch.nn as nn
import torch.nn.functional as F

'''
modified to fit dataset size from bearpaw/pytorch-classification/blob/master/models/cifar/alexnet
'''

__all__ = ['convnet_cifar'] ##edited by ffk

NUM_CLASSES = 10

class ConvNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(                              # input 32x32x3
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 16x16x64 
            nn.ReLU(inplace=True),                                  
            nn.Conv2d(64, 64, kernel_size=3, padding=1),            # 16x16x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                            # 8x8x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # 8x8x128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),          # 8x8x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                            # 4x4x128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),          # 4x4x256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # 4x4x256
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 512),                                # 4096 -> 512
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),                                    # 512 -> 512
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),                            # 512 -> 10

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def convnet_cifar(): ##edited by ffk
    model = ConvNet()
    return model
