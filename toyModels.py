import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, VisionDataset
from torch.utils.data import Dataset
import numpy as np

from genToyData import *

import torch.nn as nn
import torch.nn.functional as F

# None of these models are working.

#conv339 runs but has a skip connection from input to output
#fc2fc does not compile
#CoL not implemented


#conv(3×3×9) → avg(9×9) → fc(36×2)
class conv339(nn.Module):
    def __init__(self):
        super(conv339, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=9, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.pool = nn.AvgPool2d(kernel_size=9, stride=1, padding=0,count_include_pad=False)
        self.fc1 = nn.Linear(2*2*9, 2, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 2*2*9)
        x = self.fc1(x)
        return x


#fc(100×36)→fc(36×2)
class fc2fc(nn.Module):
    def __init__(self):
        super(fc2fc, self).__init__()
        self.fc1 = nn.Linear(10*10, 36, bias=True)
        self.fc2 = nn.Linear(36, 2, bias=True)

    def forward(self, x):
        x = x.view(-1,100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#CoL(4×4)→avg(5×5)→fc(36×2)
class ColNet(nn.Module):
    def __init__(self):
        super(ColNet, self).__init__()
        self.fc1 = nn.CoL(kernel_size, k = 4)
        self.pool = nn.AvgPool2d((2,2), stride=2, padding=0,count_include_pad=False)
        self.fc2 = nn.Linear(36, 2, bias=True)

    def forward(self, x):
        x = self.pool(self.CoL(x))
        x = x.view(-1,36)
        x = self.fc2(x)
        return x



