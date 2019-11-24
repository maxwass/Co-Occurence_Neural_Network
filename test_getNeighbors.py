import torch
import time
import numpy as np

from getNeighbors import *
fmWidth = 5
sfWidth = 3
borderSize = np.floor_divide(sfWidth,2)
# (int high, dim of tensor)
input_tensor = torch.randint(3,(2,3,fmWidth, fmWidth))
print(input_tensor)

#ensure indexing works correctly
batch, chan, row, col = 1,2,0,0
#print(input_tensor[batch,chan, row, col].item())

#create indeces
ls,rs,ts,bs=[],[],[],[]
for i in np.arange(fmWidth):
    l,r = (batch,chan,i,0), (batch,chan,i,fmWidth-1)
    t,b = (batch,chan,0,i), (batch,chan,fmWidth-1,i)
    ls.append(l)
    rs.append(r)
    ts.append(t)
    bs.append(b)

left,right,top,bot = 0,1,2,3
for p in ls:
    assert typeOfPixel(fmWidth,borderSize,p)[left], \
            "pixel {0} should be in LEFT Border".format(p)
for p in rs:
    assert typeOfPixel(fmWidth,borderSize,p)[right], \
            'pixel {0} should be in RIGHT Border'.format(p)
for p in ts:
    assert typeOfPixel(fmWidth,borderSize,p)[top], \
            'pixel {0} should be in TOP Border'.format(p)
for p in bs:
    assert typeOfPixel(fmWidth,borderSize,p)[bot], \
            'pixel {0} should be in BOTTOM Border'.format(p)

# Middle
for i in np.arange(1,fmWidth-1):
    for j in np.arange(1,fmWidth-1):
        p = (batch,chan,i,j)
        assert (not any(typeOfPixel(fmWidth,borderSize,p)),\
            "{0} should not be on border".format(p))
