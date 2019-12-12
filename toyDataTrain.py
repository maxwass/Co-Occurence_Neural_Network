import torch
from torchvision import transforms
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, VisionDataset
from torch.utils.data import Dataset
import numpy as np
from statistics import mean
import os.path

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


numChannels, fmHeight, fmWidth = 1, 10, 10
image_size = (numChannels, fmHeight, fmWidth)

pixel_vals = [0.0,0.333,0.666,1.0]
distribs   = np.array([[0.1,0.4,0.4,0.1],[0.4,0.1,0.1,0.4]])

lr = 1e-4
num_iterations, batch_size, num_workers = 2000, 2, 2
train_size, test_size, num_classes, random_offset = 6000, 1000, 2, 3
trainset    = ToyData(train_size, num_classes, image_size, distribs, random_offset, pixel_vals)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset    = ToyData(test_size, num_classes, image_size, distribs, random_offset, pixel_vals)
testloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
classes     = ('0: (.1 .4 .4 .1) ', '1: (.4 .1 .1 .4)')

input_tensor_shape = (batch_size,)+image_size


#choose network
network = "col"
if network=="col":
    PATH = './models/col_net.pth'
    w_shape = (1,3,3)
    net = ColNet(input_tensor_shape, 4, w_shape) #params 90
elif network=="conv332":
    net = conv332() #params: 86
    PATH = './models/conv332_net.pth'
elif network=="conv119":
    net = conv119() #params: 92
    PATH = './models/conv119_net.pth'
elif network=="conv339":
    net = conv339() #params: 164 (155 from my calc, must be bias in convs)
    PATH = './models/conv339_net.pth'
elif network=="fc":
    net = fc2fc()  #params: 3710
    PATH = './models/fc2fc_net.pth'


if os.path.isfile(PATH):
    net.load_state_dict(torch.load(PATH))
    net.eval()
    print (f'Model {network} Exists in {PATH}: loading old params...')
else:
    print (f'Model {network} does not exist, starting cold...')

num_net_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'# Model Parameters: {num_net_params}')
print(str(net))

#Where to train model
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print('Is GPU available?\n\t')
print(torch.cuda.is_available())


#optimizer
criterion = nn.CrossEntropyLoss() #loss not specified in paper but in code only cross entroyp
optimizer = optim.Adam(net.parameters(), lr=lr)#, momentum=0.9)


#tensorboard
# default `log_dir` is "runs" - we'll be more specific here
tensorboard_proj='runs/'+network
writer = SummaryWriter(tensorboard_proj)

# get some random training images
images, labels = next(iter(trainloader))
writer.add_graph(net,images)
writer.close()

torch.set_printoptions(precision=2)
#training loop
running_loss, best_loss = 0.0, np.inf
times = []
for epoch in range(num_iterations):  # loop over the dataset multiple times
    t0 = time.time_ns()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data #data is a list of [inputs, labels]
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if loss.item() < best_loss:
            best_loss = loss.item()
            #save params
            torch.save(net.state_dict(), PATH)
        if True and (i % 10 == 9):    # every 100 mini-batches...
            print(f'{epoch} epoch, {i}th minibatch loop')
            print(f'\tloss: {loss.item():.3}')

    #log/display after each epoch
    t1 = time.time_ns()
    epoch_time = (t1-t0)/(10**9)
    times.append(epoch_time)
    writer.add_scalar('training loss', running_loss / 100 , global_step=epoch)
    acc_train, acc_test = compute_acc(trainset,"train"), compute_acc(testset, "test")
    writer.add_scalar('test accuracy', acc_test,global_step=epoch)
    writer.add_scalar('train accuracy', acc_train,global_step=epoch)
    writer.add_scalar('epoch/s', epoch_time, global_step=epoch)
    writer.flush()
    print(f'\n{epoch} epoch')
    print(f'\trunning loss: {running_loss:.3}')
    running_loss = 0.0
    t = np.asarray(times)
    print(f'\tepoch/s: {epoch_time:.3},  mean: {np.mean(t):.3}, +- {np.std(t):.3}')
    #print(f'\ttrain acc: {acc_train:.3}, test_acc: {acc_test:.3}\n')
    #times.clear()
    """
    if(False):
        print(f'outputs: {outputs}')
        print(f'torch.max(outputs.data, 1): {torch.max(outputs.data, 1)}')
        _, pred = torch.max(outputs.data, 1)
        inspect_batch(data,outputs=pred,print_img=True)

    if i % 100 == 99:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss', running_loss / 100,global_step=j)
            running_loss = 0.0
            acc_train = compute_acc(trainset,"train")
            acc_test  = compute_acc(testset, "test")
            writer.add_scalar('test accuracy', acc_test,global_step=j)
            writer.add_scalar('train accuracy', acc_train,global_step=j)
            writer.flush()
            print(f'{epoch} epoch, {i}th minibatch loop')
            print(f'\tloss: {loss.item()}')
            t = np.asarray(times)
            print(f'\timage/time: {np.mean(t)/1000}, +- {np.std(t)/1000}')
            times.clear()
            if(False):
                print(f'outputs: {outputs}')
                print(f'torch.max(outputs.data, 1): {torch.max(outputs.data, 1)}')
                _, pred = torch.max(outputs.data, 1)
                inspect_batch(data,outputs=pred,print_img=True)
        """

print('Finished Training')

