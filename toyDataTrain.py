import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, VisionDataset
from torch.utils.data import Dataset
import numpy as np

from genToyData import *
from toyModels import *

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter
def compute_acc(dataset, name):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
    correct, total = 0, 0
    acc = 0.0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = (100 * correct / total)
        print(f'\t{name}: {correct}/{total} => {acc}')
    return acc





train_size, test_size, num_classes, image_size, random_offset = 6000, 1000, 2, (1, 10, 10), 3
pixel_vals = [0.0,0.333,0.666,1.0]
distribs   = np.array([[0.1,0.4,0.4,0.1],[0.4,0.1,0.1,0.4]])
num_iterations, batch_size, num_workers = 2000, 3, 2
lr = 1e-4

trainset    = ToyData(train_size, num_classes, image_size, distribs, random_offset, pixel_vals)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset    = ToyData(test_size, num_classes, image_size, distribs, random_offset, pixel_vals)
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
classes     = ('0: (.1 .4 .4 .1) ', '1: (.4 .1 .1 .4)')


#choose network
net = conv339() #params: 164
#net = fc2fc()  #params: 3710
num_net_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'# Model Parameters: {num_net_params}')

#optimizer
criterion = nn.CrossEntropyLoss() #loss not specified in paper but in code only cross entroyp
optimizer = optim.Adam(net.parameters(), lr=lr)#, momentum=0.9)


#tensorboard

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/toyData_exp_2')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
writer.add_graph(net, images)
writer.close()

torch.set_printoptions(precision=2)
#training loop
running_loss = 0.0
j = 0
for epoch in range(num_iterations):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 99:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss', running_loss / 1000,global_step=j)# epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
            #TODO
            #writer.add_figure('predictions vs. actuals', plot_classes_preds(net, inputs, labels), global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
            print(f'{epoch} epoch, {i}th minibatch loop')
            acc_train = compute_acc(trainset,"train")
            acc_test  = compute_acc(testset, "test")
            writer.add_scalar('test accuracy', acc_test,global_step=j)
            writer.add_scalar('train accuracy', acc_train,global_step=j)
            if(False):
                print(f'outputs: {outputs}')
                print(f'torch.max(outputs.data, 1): {torch.max(outputs.data, 1)}')
                _, pred = torch.max(outputs.data, 1)
                inspect_batch(data,outputs=pred,print_img=True)
        j+=1

print('Finished Training')

