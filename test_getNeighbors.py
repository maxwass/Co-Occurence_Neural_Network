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

"""
## test if borderLocPixel correctly identifies where on border pixel is
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
"""

#inChannel Neighbors
for p in ls:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    col = p[3]
    c = (colLims==[0,col+borderSize])
    assert c ,\
            f'wrong 2D neighbor calc: LEFT pixel {p}, bS:{borderSize}:\
            \n\t\t\tcolLims: {colLims}'
    #print("\n(row,col): ({0},{1})".format(p[2],p[3]))
    #print("colLims: {0} -> {1}".format(colLims[0], colLims[1]))
for p in rs:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    col = p[3]
    c = (colLims==[col-borderSize,fmWidth-1])
    assert c ,\
            f'wrong 2D neighbor calc: RIGHT pixel {p}, bS:{borderSize}:\
            \n\t\t\tcolLims: {colLims}'
    #print("\n(row,col): ({0},{1})".format(p[2],p[3]))
    #print("colLims: {0} -> {1}".format(colLims[0], colLims[1]))

for p in ts:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    row = p[2]
    r = (rowLims==[0,row+borderSize])
    assert r ,\
            f'wrong 2D neighbor calc: TOP pixel {p}, bS:{borderSize}:\
            \n\t\t\trowLims: {rowLims}'
    #print("\n(row,col): ({0},{1})".format(p[2],p[3]))
    #print("rowLims: {0} -> {1}".format(colLims[0], colLims[1]))

for p in bs:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    row = p[2]
    r = (rowLims==[row-borderSize,fmWidth-1])
    assert r ,\
            f'wrong 2D neighbor calc: TOP pixel {p}, bS:{borderSize}:\
            \n\t\t\trowLims: {rowLims}'
    #print("\n(row,col): ({0},{1})".format(p[2],p[3]))
    #print("rowLims: {0} -> {1}".format(colLims[0], colLims[1]))

for p in mdl:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    row,col = p[2],p[3]
    r = (rowLims==[row-borderSize,row+borderSize])
    c = (colLims==[col-borderSize,col+borderSize])
    assert r and c ,\
            f'wrong 2D neighbor calc: MIDDLE pixel {p}, bS:{borderSize}:\
            \n\t\t\trowLims:{rowLims}, colLims: {colLims}'
    #print("\n(row,col): ({0},{1})".format(p[2],p[3]))
    #print("rowLims: {0} -> {1}".format(rowLims[0], rowLims[1]))
    #print("colLims: {0} -> {1}".format(colLims[0], colLims[1]))

