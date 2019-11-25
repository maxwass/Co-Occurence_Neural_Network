import torch
import time
import numpy as np

from getNeighbors import *
fmWidth = 8
sfWidth = 5
sfDepth = 3
borderSize = np.floor_divide(sfWidth,2)
# (int high, dim of tensor)
input_tensor = torch.randint(3,(2,3,fmWidth, fmWidth))
print(input_tensor)

#ensure indexing works correctly
batch, chan, row, col = 1,2,0,0
#print(input_tensor[batch,chan, row, col].item())

#create indeces: lefts, rights, top, bottoms, middle(non border)
ls,rs,ts,bs,mdl=[],[],[],[],[]
for i in np.arange(fmWidth):
    for j in range(borderSize):
        l,r = (batch,chan,i,j), (batch,chan,i,fmWidth-1-j)
        t,b = (batch,chan,j,i), (batch,chan,fmWidth-1-j,i)
        ls.append(l)
        rs.append(r)
        ts.append(t)
        bs.append(b)

for i in np.arange(borderSize,fmWidth-1-borderSize):
    for j in np.arange(borderSize,fmWidth-1-borderSize):
        m = (batch,chan,i,j)
        mdl.append(m)

## borderLocPixel
left,right,top,bot = 0,1,2,3
for p in ls:
    assert borderLocPixel(fmWidth,borderSize,p)[left], \
            "pixel {0} should be in LEFT Border".format(p)
for p in rs:
    assert borderLocPixel(fmWidth,borderSize,p)[right], \
            'pixel {0} should be in RIGHT Border'.format(p)
for p in ts:
    assert borderLocPixel(fmWidth,borderSize,p)[top], \
            'pixel {0} should be in TOP Border'.format(p)
for p in bs:
    assert borderLocPixel(fmWidth,borderSize,p)[bot], \
            'pixel {0} should be in BOTTOM Border'.format(p)
# Middle
for p in mdl:
    onBorder = any(borderLocPixel(fmWidth,borderSize,p))
    assert not onBorder, "{0} should not be on border".format(p)

#inChannelNeighbors
for p in ls:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    col = p[3]
    c = (colLims==[0,col+borderSize])
    assert c ,\
            f'wrong 2D neighbor calc: LEFT pixel {p}, bS:{borderSize}:\
            \n\t\t\tcolLims: {colLims}'
for p in rs:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    col = p[3]
    c = (colLims==[col-borderSize,fmWidth-1])
    assert c ,\
            f'wrong 2D neighbor calc: RIGHT pixel {p}, bS:{borderSize}:\
            \n\t\t\tcolLims: {colLims}'

for p in ts:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    row = p[2]
    r = (rowLims==[0,row+borderSize])
    assert r ,\
            f'wrong 2D neighbor calc: TOP pixel {p}, bS:{borderSize}:\
            \n\t\t\trowLims: {rowLims}'

for p in bs:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    row = p[2]
    r = (rowLims==[row-borderSize,fmWidth-1])
    assert r ,\
            f'wrong 2D neighbor calc: TOP pixel {p}, bS:{borderSize}:\
            \n\t\t\trowLims: {rowLims}'

for p in mdl:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    row,col = p[2],p[3]
    r = (rowLims==[row-borderSize,row+borderSize])
    c = (colLims==[col-borderSize,col+borderSize])
    assert r and c ,\
            f'wrong 2D neighbor calc: MIDDLE pixel {p}, bS:{borderSize}:\
            \n\t\t\trowLims:{rowLims}, colLims: {colLims}'


#neighborChannels
#invariant: sfDepth <= numChannels
numChannels, sfDepth = 3, 3
nCs = neighborChannels(numChannels, sfDepth, (0,0,0,0))
assert ([2,0,1]==nCs).all(), \
        f'output channel is 0th one'
nCs = neighborChannels(numChannels, sfDepth, (0,1,0,0))
assert ([0,1,2]==nCs).all(), \
        f'output channel is middle'
nCs = neighborChannels(numChannels, sfDepth, (0,2,0,0))
assert ([1,2,0]==nCs).all(), \
        f'output channel is (numChan-1)th one'

numChannels, sfDepth = 8, 3
nCs = neighborChannels(numChannels, sfDepth, (0,0,0,0))
assert ([7,0,1]==nCs).all(), \
         f'output channel is 0th of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,1,0,0))
assert ([0,1,2]==nCs).all(), \
        f'output channel is 1st of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,2,0,0))
assert ([1,2,3]==nCs).all(), \
        f'output channel is 2nd of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,3,0,0))
assert ([2,3,4]==nCs).all(), \
        f'output channel is 3rd of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,5,0,0))
assert ([4,5,6]==nCs).all(), \
        f'output channel is 5th of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,6,0,0))
assert ([5,6,7]==nCs).all(), \
        f'output channel is 6th of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,7,0,0))
assert ([6,7,0]==nCs).all(), \
        f'output channel is 7th of 7'


numChannels, sfDepth = 8, 5
nCs = neighborChannels(numChannels, sfDepth, (0,0,0,0))
assert ([6,7,0,1,2]==nCs).all(), \
         f'output channel is 0th of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,1,0,0))
assert ([7,0,1,2,3]==nCs).all(), \
        f'output channel is 1st of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,2,0,0))
assert ([0,1,2,3,4]==nCs).all(), \
        f'output channel is 2nd of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,3,0,0))
assert ([1,2,3,4,5]==nCs).all(), \
        f'output channel is 3rd of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,5,0,0))
assert ([3,4,5,6,7]==nCs).all(), \
        f'output channel is 5th of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,6,0,0))
assert ([4,5,6,7,0]==nCs).all(), \
        f'output channel is 6th of 7'
nCs = neighborChannels(numChannels, sfDepth, (0,7,0,0))
assert ([5,6,7,0,1]==nCs).all(), \
        f'output channel is 7th of 7'


numChannels, sfDepth = 128, 3
nCs = neighborChannels(numChannels, sfDepth, (0,0,0,0))
assert ([127,0,1]==nCs).all(), \
         f'output channel is 0th of 127'
nCs = neighborChannels(numChannels, sfDepth, (0,1,0,0))
assert ([0,1,2]==nCs).all(), \
        f'output channel is 1st of 127'
nCs = neighborChannels(numChannels, sfDepth, (0,2,0,0))
assert ([1,2,3]==nCs).all(), \
        f'output channel is 2nd of 127'
nCs = neighborChannels(numChannels, sfDepth, (0,125,0,0))
assert ([124,125,126]==nCs).all(), \
        f'output channel is 125th of 127'
nCs = neighborChannels(numChannels, sfDepth, (0,126,0,0))
assert ([125,126,127]==nCs).all(), \
        f'output channel is 126th of 127'
nCs = neighborChannels(numChannels, sfDepth, (0,127,0,0))
assert ([126,127,0]==nCs).all(), \
        f'output channel is 127th of 127'



#getNeighborhoodIndeces
fmWidth,numChannels,sfDims = 8,128,(3,3,3) #Height, Width, Depth
chan = 1
wNieghbors, nieghbors = getNeighborhoodIndeces(fmWidth, numChannels, sfDims, (0,chan,0,0))
print(wNieghbors)
print(nieghbors)
chan = 1
#print(getNeighborhoodIndeces(fmWidth, numChannels, sfDims, (0,chan,0,0)))
