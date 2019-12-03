import torch
#conda install -c conda-forge torchvision
#from torchvision import vision
#from vision import VisionDataset
#import transforms
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, VisionDataset
#from .vision import VisionDataset
#from .. import transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ToyData(VisionDataset):
    """A fake dataset that returns randomly generated images

    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (1,10,10)
        num_classes(int, optional): Number of classes in the datset. Default: 2

    """

    def __init__(self, size, num_classes, image_size, distribs, random_offset=0, pixel_vals = [0.0,0.25,0.5,1.0]):
        super(ToyData, self).__init__(None, transform=None, target_transform=None)
        #super(VisionDataset, self).__init__(None)
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.pixel_vals = pixel_vals
        self.random_offset = random_offset
        self.distribs = distribs

        self.__getInfo__()
        """
        torch.set_printoptions(precision=2)
        for i in range(size):
            img, target = self.__getitem__(i)
            print(f'target: {target}, distrib:{self.distribs[target,:]}')
            print(img)
            input("generate next data item...")
        torch.set_printoptions(precision=5)
        """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))

        # flip coin seed init
        target = np.random.randint(0,self.num_classes)

        #get target distrib and create image
        target_distrib = self.distribs[target,:]
        img = np.random.choice(self.pixel_vals, self.image_size, p=target_distrib)
        #print(f'target: {target},target_distrib: {target_distrib}')

        return img, target

    def __len__(self):
        return self.size
    def __pixel_vals__(self):
        return self.pixel_vals
    def __image_size__(self):
        return self.image_size
    def __num_classes__(self):
        return self.num_classes
    def __distribs__(self):
        return self.distribs
    def __getInfo__(self):
        print(f'dataset object:')
        print(f'\tsize: {self.size}, image_size:{self.image_size}, num_classes: {self.num_classes}')
        print(f'\tpixel_vals: {self.pixel_vals}')
        print(f'\tdistributions:')
        for i in range(num_classes):
            print(f'\t\t{i}: {self.distribs[i,:]}')

size, num_classes, image_size, random_offset = 6000, 2, (1, 10, 10), 0
pixel_vals = [0.0,0.25,0.5,1.0]
distribs = np.array([[0.1,0.4,0.4,0.1,],[0.4,0.1,0.1,0.4]])
dataset = ToyData(size, num_classes, image_size, distribs, random_offset, pixel_vals)

batchSize, numWorkers = 5, 2
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize,shuffle=True, num_workers=numWorkers)

