import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, VisionDataset
from torch.utils.data import Dataset
import numpy as np

from genToyData import *
from col import *
from lookupTable import genLookUpTable

import torch.nn as nn
import torch.nn.functional as F

#conv(3×3×9) → avg(9×9) → fc(36×2)
class conv339(nn.Module):
    def __init__(self):
        super(conv339, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,out_channels=9, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.pool = nn.AvgPool2d(kernel_size=9, stride=1, padding=0,count_include_pad=False)
        self.fc = nn.Linear(2*2*9, 2, bias=True)

    def forward(self, x):
        #print(f'input shape: {x.size()}') #[3, 1, 10, 10]
        x = F.relu(self.conv(x))
        #print(f'after conv shape: {x.size()}') #[3, 9, 10, 10]
        x = self.pool(x)
        #print(f'after pool shape: {x.size()}') #[3, 9, 2, 2]
        x = x.view(-1, 2*2*9)
        #print(f'after resize shape: {x.size()}') #[3, 36]
        x = self.fc(x)
        #print(f'after fc shape: {x.size()}') #[3, 2]
        #input('..check sizes..')
        return x


#fc(100×36)→fc(36×2)
class fc2fc(nn.Module):
    def __init__(self):
        super(fc2fc, self).__init__()
        self.fc1 = nn.Linear(10*10, 36, bias=True)
        self.fc2 = nn.Linear(36, 2, bias=True)

    def forward(self, x):
        #print(f'\ninput shape: {x.size()}\n') #[3, 1, 10, 10]
        x = x.view(-1,100)
        #print(f'\nafter reshape: {x.size()}\n') #[3, 100]
        x = F.relu(self.fc1(x))
        #print(f'\nafter fc1111 shape: {x.size()}\n') #[3, 36]
        x = self.fc2(x)
        #print(f'\nafter fc2222 shape: {x.size()}\n') #[3, 2]
        #input('..check sizes..')
        return x

class CoL_Module(nn.Module):                                           #HxWxD
    def __init__(self, input_tensor_size, co_shape=(5,5), w_shape=(1,3,3), learn_co=True, learn_w=True):
        super(CoL_Module, self).__init__()

        """print(f'IN COL __INIT__: ABOUT TO UNPACK INPUT_TENSOR_SIZE INTO 4:')
        print(f'type(input_tensor_size): {type(input_tensor_size)}')
        input('printing entire input_tensor_size')
        print(f'{input_tensor_size}')
        input('inspect input_tensor_size')
        """
        numBatches,numChannels,fmHeight,fmWidth = input_tensor_size

        assert all(x>0 for x in input_tensor_size) and fmHeight==fmWidth
        assert co_shape[0]==co_shape[1] and isinstance(co_shape[0],int)\
                and all(x>0 for x in co_shape), \
                'CoL_Module given improper shape for cooccurence matrix: {co_shape}'

        (w_depth, w_height,w_width) = w_shape
        print(f'spatial filter W dims: ({w_height}, {w_width}, {w_depth})')
        assert w_height==w_width and \
                isinstance(w_height,int) and \
                isinstance(w_depth,int) and \
                all(x>0 for x in w_shape), \
                f'CoL_Module given improper shape for spatial filter w: {w_shape}'
        assert learn_co or learn_w, f'CoL_Module must learn L {learn_co} or W {learn_w}'

        self.k                 = co_shape[0]
        self.w_shape           = w_shape
        self.learn_co          = learn_co
        self.learn_w           = learn_w
        self.input_tensor_size = input_tensor_size
        self.neighbor_lookup_table = genLookUpTable(input_tensor_size, (w_width,w_width,w_depth)) #must be square in 2D
        self.actBounds         = (0.0,1.0) #inputs are limited to be between 0 and 1

        if learn_co:
            self.L = nn.Parameter((torch.normal(mean=0.0, std=.1,size=(self.k,self.k))))
            #self.L = nn.Parameter((torch.ones( (self.k,self.k), dtype=torch.float32)))
            #nn.init.normal_(L, mean=0.0, std=.1) #TODO truncate
        else:
            self.L = torch.ones(size=(self.k,self.k), dtype=torch.float32)
            self.register_buffer("Deep Co-Occur L", self.L)

        if learn_w:
            self.W = nn.Parameter(torch.normal(mean=0.0, std=.1,size=(w_depth,w_height,w_width))) #TODO truncate
        else:
            self.W = torch.ones(size=self.w_shape, dtype=torch.float32)
            self.register_buffer("Spatial Filter W", self.W)

    def forward(self, input_tensor):
        # See the autograd section for explanation of what happens here.
        # TO DO normalize input between (0,1)
        #print(f'IN COL FORWARD: {input_tensor.size()}')
        #input('...')
        return CoL(input_tensor, self.W, self.L, self.actBounds, self.neighbor_lookup_table)

    def extra_repr(self):
        return f'L=({self.k},{self.k}), W={self.w_shape}'

    def getInfo(self):
        # Extra information about this module. Access by printing an object
        # of this class.
        lc ='IS' if self.learn_co else 'IS NOT'
        lw ='IS' if self.learn_co else 'IS NOT'

        spiel  = f'CoL_Module: \n\t'
        spiel += f'L is of size {self.k}x{self.k}.  L {lc} being learned.\n\t'
        spiel += f'W dims (H,W,D)= {self.w_shape}.  W {lw} being learned.\n\t'
        spiel += f'Expecting input tensor (batch,chan,height,width) = {self.input_tensor_size}\n\t'
        spiel += f'Current W vals: \n\t{self.W}\n\t'
        spiel += f'Current L vals: \n\t{self.L}\n\t'
        return spiel

#CoL(4×4)→avg(5×5)→fc(36×2)
class ColNet(nn.Module):
    def __init__(self, input_tensor_size):
        super(ColNet, self).__init__()                          #DepthxHeightxWidth
        self.col = CoL_Module(input_tensor_size, co_shape=(4,4), w_shape=(1,3,3), learn_w=False)
        self.pool = nn.AvgPool2d((5,5), stride=1, padding=0,count_include_pad=False)
        self.fc = nn.Linear(36, 2, bias=True)

    def forward(self, x):
        #print(f'input shape: {x.size()}') #[3, 1, 10, 10]
        """
        print('IN ColNet FORWARD: about to step into self.col')
        print(f'x.size(): {x.size()}')
        input('about to print x itself...')
        print(x)
        input('...inspect x')
        """
        x = self.pool(self.col(x))
        x = x.view(-1,36)
        x = self.fc(x)
        return x
