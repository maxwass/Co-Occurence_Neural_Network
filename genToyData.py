import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, VisionDataset
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

    def __init__(self, size, num_classes, image_size, distribs, random_offset=0, pixel_vals = [0.0,0.333,0.666,1.0]):
        super(ToyData, self).__init__(None, transform=None, target_transform=None)
        #super(VisionDataset, self).__init__(None)
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.pixel_vals = pixel_vals
        self.random_offset = random_offset
        self.distribs = distribs
        self.dataset = []
        self.__getInfo__()
        self.__genData__()

    # generate all the data now and save in list for indexing later
    def __genData__(self):
        numOnes = 0
        for i in range(self.size):
            rng_state = torch.get_rng_state()

            torch.manual_seed(i + self.random_offset)
            target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
            numOnes +=target
            target_distrib = self.distribs[target,:]
            img = np.random.choice(self.pixel_vals, self.image_size, p=target_distrib)

            torch.set_rng_state(rng_state)
            img = torch.tensor(img, dtype=torch.float32)
            #img = transforms.ToPILImage()(img)

            self.dataset.append((img,target))

        print(f'Dataset Ratio: 0s {self.size-numOnes}: 1s {numOnes}')

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
        return self.dataset[index]

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
        for i in range(self.num_classes):
            print(f'\t\t{i}: {self.distribs[i,:]}')


def inspect_batch(data,outputs=None, print_img=False):
    samples, targets = data
    n = targets.size()[0]
    for i, sample in enumerate(samples):
        l = f' target {targets[i]}'
        if outputs is not None:
            l+=f', network pred {outputs[i]} '

        print(f'\tsample {i+1}/{n} in batch:'+l)
        if(print_img):
            print(f'\tSize of sample input: {sample.size()}')
            print(f'\t{sample[0,:,:]}\n')

"""
train_size, test_size, num_classes, image_size, random_offset = 6000, 2000, 2, (1, 10, 10), 0
pixel_vals = [0.0,0.333,0.666,1.0]
distribs = np.array([[0.1,0.4,0.4,0.1],[0.4,0.1,0.1,0.4]])
batch_size, num_workers = 3, 2

#FROM PAPER CODE
lr = 1e-4
trainset    = ToyData(train_size, num_classes, image_size, distribs, random_offset, pixel_vals)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset    = ToyData(test_size, num_classes, image_size, distribs, random_offset, pixel_vals)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
classes    = ('0: .1 .4 .4 .1', '1: .4 .1 .1 .4')




torch.set_printoptions(precision=2)
for i, sample_batched in enumerate(trainloader,0):
    data, targets = sample_batched
    inspect_batch(sample_batched)

"""
