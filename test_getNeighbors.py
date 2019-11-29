import torch
import time
import numpy as np

from getNeighbors import *

def genBorderIndeces(fmWidth, borderSize):
    ls,rs,ts,bs,mdl=[],[],[],[],[]
    for i in np.arange(fmWidth):
        for j in range(borderSize):
            l,r = (i,j), (i,fmWidth-1-j)
            t,b = (j,i), (fmWidth-1-j,i)
            ls.append(l)
            rs.append(r)
            ts.append(t)
            bs.append(b)

    for i in np.arange(borderSize,fmWidth-1-borderSize):
        for j in np.arange(borderSize,fmWidth-1-borderSize):
            m = (i,j)
            mdl.append(m)

    return ls,rs,ts,bs,mdl


fmWidth, sfWidth, sfDepth  = 8,5,3
borderSize = np.floor_divide(sfWidth,2)

ls,rs,ts,bs,mdl= genBorderIndeces(fmWidth, borderSize)

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
    col = p[1]
    c = (colLims==[0,col+borderSize])
    assert c ,\
            f'wrong 2D neighbor calc: LEFT pixel {p}, bS:{borderSize}:\
            \n\t\t\tcolLims: {colLims}'
for p in rs:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    col = p[1]
    c = (colLims==[col-borderSize,fmWidth-1])
    assert c ,\
            f'wrong 2D neighbor calc: RIGHT pixel {p}, bS:{borderSize}:\
            \n\t\t\tcolLims: {colLims}'

for p in ts:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    row = p[0]
    r = (rowLims==[0,row+borderSize])
    assert r ,\
            f'wrong 2D neighbor calc: TOP pixel {p}, bS:{borderSize}:\
            \n\t\t\trowLims: {rowLims}'

for p in bs:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    row = p[0]
    r = (rowLims==[row-borderSize,fmWidth-1])
    assert r ,\
            f'wrong 2D neighbor calc: TOP pixel {p}, bS:{borderSize}:\
            \n\t\t\trowLims: {rowLims}'

for p in mdl:
    rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth, p)
    row,col = p[0],p[1]
    r = (rowLims==[row-borderSize,row+borderSize])
    c = (colLims==[col-borderSize,col+borderSize])
    assert r and c ,\
            f'wrong 2D neighbor calc: MIDDLE pixel {p}, bS:{borderSize}:\
            \n\t\t\trowLims:{rowLims}, colLims: {colLims}'


#neighborChannels
#invariant: sfDepth <= numChannels
numChannels, sfDepth = 3, 3
nCs = neighborChannels(numChannels, sfDepth, 0)
assert ([2,0,1]==nCs).all(), \
        f'output channel is 0th one'
nCs = neighborChannels(numChannels, sfDepth, 1)
assert ([0,1,2]==nCs).all(), \
        f'output channel is middle'
nCs = neighborChannels(numChannels, sfDepth, 2)
assert ([1,2,0]==nCs).all(), \
        f'output channel is (numChan-1)th one'

numChannels, sfDepth = 8, 3
nCs = neighborChannels(numChannels, sfDepth, 0)
assert ([7,0,1]==nCs).all(), \
         f'output channel is 0th of 7'
nCs = neighborChannels(numChannels, sfDepth, 1)
assert ([0,1,2]==nCs).all(), \
        f'output channel is 1st of 7'
nCs = neighborChannels(numChannels, sfDepth, 2)
assert ([1,2,3]==nCs).all(), \
        f'output channel is 2nd of 7'
nCs = neighborChannels(numChannels, sfDepth, 3)
assert ([2,3,4]==nCs).all(), \
        f'output channel is 3rd of 7'
nCs = neighborChannels(numChannels, sfDepth, 5)
assert ([4,5,6]==nCs).all(), \
        f'output channel is 5th of 7'
nCs = neighborChannels(numChannels, sfDepth, 6)
assert ([5,6,7]==nCs).all(), \
        f'output channel is 6th of 7'
nCs = neighborChannels(numChannels, sfDepth, 7)
assert ([6,7,0]==nCs).all(), \
        f'output channel is 7th of 7'


numChannels, sfDepth = 8, 5
nCs = neighborChannels(numChannels, sfDepth, 0)
assert ([6,7,0,1,2]==nCs).all(), \
         f'output channel is 0th of 7'
nCs = neighborChannels(numChannels, sfDepth, 1)
assert ([7,0,1,2,3]==nCs).all(), \
        f'output channel is 1st of 7'
nCs = neighborChannels(numChannels, sfDepth, 2)
assert ([0,1,2,3,4]==nCs).all(), \
        f'output channel is 2nd of 7'
nCs = neighborChannels(numChannels, sfDepth, 3)
assert ([1,2,3,4,5]==nCs).all(), \
        f'output channel is 3rd of 7'
nCs = neighborChannels(numChannels, sfDepth, 5)
assert ([3,4,5,6,7]==nCs).all(), \
        f'output channel is 5th of 7'
nCs = neighborChannels(numChannels, sfDepth, 6)
assert ([4,5,6,7,0]==nCs).all(), \
        f'output channel is 6th of 7'
nCs = neighborChannels(numChannels, sfDepth, 7)
assert ([5,6,7,0,1]==nCs).all(), \
        f'output channel is 7th of 7'


numChannels, sfDepth = 128, 3
nCs = neighborChannels(numChannels, sfDepth, 0)
assert ([127,0,1]==nCs).all(), \
         f'output channel is 0th of 127'
nCs = neighborChannels(numChannels, sfDepth, 1)
assert ([0,1,2]==nCs).all(), \
        f'output channel is 1st of 127'
nCs = neighborChannels(numChannels, sfDepth, 2)
assert ([1,2,3]==nCs).all(), \
        f'output channel is 2nd of 127'
nCs = neighborChannels(numChannels, sfDepth, 125)
assert ([124,125,126]==nCs).all(), \
        f'output channel is 125th of 127'
nCs = neighborChannels(numChannels, sfDepth, 126)
assert ([125,126,127]==nCs).all(), \
        f'output channel is 126th of 127'
nCs = neighborChannels(numChannels, sfDepth, 127)
assert ([126,127,0]==nCs).all(), \
        f'output channel is 127th of 127'


fmWidth,numChannels,sfDims = 8,5,(3,3,3) #Height, Width, Depth
borderSize = np.floor_divide(3,2)
ls,rs,ts,bs,mdl = genBorderIndeces(fmWidth, borderSize)

#corners
left_top  = set(ls)&set(ts)
left_bot  = set(ls)&set(bs)
right_top = set(rs)&set(ts)
right_bot = set(rs)&set(bs)

ls_only = set(ls)-left_top-left_bot
rs_only = set(rs)-right_top-right_bot
ts_only = set(ts)-left_top-right_top
bs_only = set(bs)-left_bot-right_bot

# 3 cases:
#(1) fmWidth=5, sfWidth=3
#(2) fmWidth=8, sfWidth=3
#(3) fmWidth=8, sfWidth=5

a = []
fmWidth, sfWidth = 5 , 3
            #(point, true sf neighbor indexes)

middleSFNeighbors = []
for i in range(sfWidth):
    for j in range(sfWidth):
        middleSFNeighbors.append((i,j))

test_ps = [\
          [(0,0),set([(1,1),(1,2),(2,1),(2,2)])],\
          [(1,0),set([(0,1),(0,2),(1,1),(1,2),(2,1),(2,2)])],\
          [(4,0),set([(0,1),(0,2),(1,1),(1,2)])],\
          [(0,4),set([(1,0),(1,1),(2,0),(2,1)])],\
          [(3,4),set([(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)])],\
          [(4,4),set([(0,0),(0,1),(1,0),(1,1)])],\
          [(4,2),set([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)])],\
          [(0,1),set([(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)])],\
          [(2,2),set(middleSFNeighbors)]\
          ]
a.append((fmWidth,sfWidth,test_ps))
fmWidth, sfWidth = 8 , 3
            #(point, true sf neighbor indexes)
middleSFNeighbors = []
for i in range(sfWidth):
    for j in range(sfWidth):
        middleSFNeighbors.append((i,j))

test_ps = [\
          [(0,0),set([(1,1),(1,2),(2,1),(2,2)])],\
          [(5,0),set([(0,1),(0,2),(1,1),(1,2),(2,1),(2,2)])],\
          [(7,0),set([(0,1),(0,2),(1,1),(1,2)])],\
          [(0,7),set([(1,0),(1,1),(2,0),(2,1)])],\
          [(1,7),set([(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)])],\
          [(7,7),set([(0,0),(0,1),(1,0),(1,1)])],\
          [(7,3),set([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)])],\
          [(0,6),set([(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)])],\
          [(3,3),set(middleSFNeighbors)]\
          ]
a.append((fmWidth,sfWidth,test_ps))

fmWidth, sfWidth = 8 , 5
            #(point, true sf neighbor indexes)

middleSFNeighbors = []
for i in range(sfWidth):
    for j in range(sfWidth):
        middleSFNeighbors.append((i,j))

test_ps = [\
          #LEFT&TOP
          [(0,0),set([(2,2),(2,3),(2,4), (3,2),(3,3),(3,4), (4,2),(4,3),(4,4)])],\
          [(0,1),set([(2,1),(2,2),(2,3),(2,4), (3,1),(3,2),(3,3),(3,4), (4,1),(4,2),(4,3),(4,4)])],\
          [(1,0),set([(1,2),(1,3),(1,4), (2,2),(2,3),(2,4), (3,2),(3,3),(3,4), (4,2),(4,3),(4,4)])],\
          [(1,1),set([(1,1),(1,2),(1,3),(1,4), (2,1),(2,2),(2,3),(2,4), (3,1),(3,2),(3,3),(3,4), (4,1),(4,2),(4,3),(4,4)])],\
          #LEFT
          [(2,0),set([(0,2),(0,3),(0,4), (1,2),(1,3),(1,4), (2,2),(2,3),(2,4), (3,2),(3,3),(3,4), (4,2),(4,3),(4,4)])],\
          [(5,1),set([(0,1),(0,2),(0,3),(0,4), (1,1),(1,2),(1,3),(1,4), (2,1),(2,2),(2,3),(2,4), (3,1),(3,2),(3,3),(3,4), (4,1),(4,2),(4,3),(4,4)])],\
          #LEFT&BOTTOM
          [(6,0),set([(0,2),(0,3),(0,4), (1,2),(1,3),(1,4), (2,2),(2,3),(2,4), (3,2),(3,3),(3,4)])],\
          [(6,1),set([(0,1),(0,2),(0,3),(0,4), (1,1),(1,2),(1,3),(1,4), (2,1),(2,2),(2,3),(2,4), (3,1),(3,2),(3,3),(3,4)])],\
          [(7,0),set([(0,2),(0,3),(0,4), (1,2),(1,3),(1,4), (2,2),(2,3),(2,4)])],\
          [(7,1),set([(0,1),(0,2),(0,3),(0,4), (1,1),(1,2),(1,3),(1,4), (2,1),(2,2),(2,3),(2,4)])],\
          #BOTTOM
          [(6,2),set([(0,0),(0,1),(0,2),(0,3),(0,4), (1,0),(1,1),(1,2),(1,3),(1,4), (2,0),(2,1),(2,2),(2,3),(2,4), (3,0),(3,1),(3,2),(3,3),(3,4)])],\
          [(7,5),set([(0,0),(0,1),(0,2),(0,3),(0,4), (1,0),(1,1),(1,2),(1,3),(1,4), (2,0),(2,1),(2,2),(2,3),(2,4)])],\
          #RIGHT&BOTTOM
          [(7,6),set([(0,0),(0,1),(0,2),(0,3), (1,0),(1,1),(1,2),(1,3), (2,0),(2,1),(2,2),(2,3)])],\
          [(7,7),set([(0,0),(0,1),(0,2), (1,0),(1,1),(1,2), (2,0),(2,1),(2,2)])],\
          [(6,6),set([(0,0),(0,1),(0,2),(0,3), (1,0),(1,1),(1,2),(1,3), (2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3)])],\
          [(6,7),set([(0,0),(0,1),(0,2), (1,0),(1,1),(1,2), (2,0),(2,1),(2,2), (3,0),(3,1),(3,2)])],\
          #RIGHT
          [(5,6),set([(0,0),(0,1),(0,2),(0,3), (1,0),(1,1),(1,2),(1,3), (2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3), (4,0),(4,1),(4,2),(4,3)])],\
          [(2,7),set([(0,0),(0,1),(0,2), (1,0),(1,1),(1,2), (2,0),(2,1),(2,2), (3,0),(3,1),(3,2), (4,0),(4,1),(4,2)])],\
          #RIGHT&TOP
          [(0,6),set([(2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3), (4,0),(4,1),(4,2),(4,3)])],\
          [(0,7),set([(2,0),(2,1),(2,2), (3,0),(3,1),(3,2), (4,0),(4,1),(4,2)])],\
          [(1,6),set([(1,0),(1,1),(1,2),(1,3), (2,0),(2,1),(2,2),(2,3),(3,0),(3,1),(3,2),(3,3), (4,0),(4,1),(4,2),(4,3)])],\
          [(1,7),set([(1,0),(1,1),(1,2), (2,0),(2,1),(2,2), (3,0),(3,1),(3,2), (4,0),(4,1),(4,2)])],\
          #TOP
          [(0,2),set([(2,0),(2,1),(2,2),(2,3),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4), (4,0),(4,1),(4,2),(4,3),(4,4)])],\
          [(1,5),set([(1,0),(1,1),(1,2),(1,3),(1,4), (2,0),(2,1),(2,2),(2,3),(2,4), (3,0),(3,1),(3,2),(3,3),(3,4), (4,0),(4,1),(4,2),(4,3),(4,4)])],\
          #ADD MIDDLE CASE
          [(3,4),set(middleSFNeighbors)],\
          [(5,5),set(middleSFNeighbors)],\
          [(4,5),set(middleSFNeighbors)],\
          ]

a.append((fmWidth,sfWidth,test_ps))


for (fmWidth, sfWidth, test_ps) in a:
    for i in range(len(test_ps)):
        rowLims, colLims = inChannelNeighbors(fmWidth, sfWidth,test_ps[i][0])
        fmNeighborsIndxs = lims2Coord(rowLims,colLims)
        sfNeighborsIndxs = fm2sf(fmNeighborsIndxs, sfWidth, test_ps[i][0])
        if(False):
            print(f'p = {test_ps[i]}')
            print(f'rowLims: {rowLims}')
            print(f'colLims: {colLims}')
            input("press to contin...")
            print(f'lims2Coords: {fmNeighborsIndxs}')
            print(f'spatial filter coordinates: {sfNeighborsIndxs}')
            input("press to contin...")
        assert set(sfNeighborsIndxs)==test_ps[i][1],\
                f'sf coordinate neighbors for {test_ps[i][0]} incorrect. \
                \n\tfmWidth: {fmWidth}, sfWidth: {sfWidth}\
                \n\tShould be: {test_ps[i][1]}\
                \n\tOutput is: {sfNeighborsIndxs}\n'



#apply filter: OC[p] = sum over q in N(p): W[q]*L[quant(p),quant(q)]*IC[q]
t = torch.rand(3,3)
print(t.size()[0])
#(height,width) = t.size
numBins = 5
bins = np.arange(0,1,1/numBins)
print(bins)
quant = np.digitize(t,bins) -1
print(t)
print(quant)
print(type(quant))


def profilingPytorchGradients(numTrials):

    fmax_value = np.floor(np.finfo(np.float64).max)
    imax_value = np.iinfo(np.uint64).max
    loops = 1000000
    if(False):
        print(f'{loops} loops. add vs += vs add_')

        #torch.add
        adding   = np.zeros(numTrials)
        backward = np.zeros(numTrials)
        for t in range(len(adding)):
            param = torch.ones(1,dtype=torch.float64, requires_grad=True)
            value = torch.ones(1, dtype=torch.float64)
            t0 = time.time()
            for i in range(loops):
                value=torch.add(value,param)
            t1 = time.time()
            value.backward()
            t2 = time.time()
            adding[t]   = t1-t0
            backward[t] = t2-t1
            print(f'iter {t}, time add:{adding[t]}, back:{backward[t]}')
            #print(f'\tparam.grad: {param.grad.item()}')
        print(f'Ave Time/Stdv torch.add:  {np.mean(adding)}, {np.std(adding)}')
        print(f'Ave Time/Stdv backward:   {np.mean(backward)}, {np.std(backward)}\n\n')

        #torch.add_
        adding   = np.zeros(numTrials)
        backward = np.zeros(numTrials)
        for t in range(len(adding)):
            param = torch.ones(1,dtype=torch.float64, requires_grad=True)
            value = torch.ones(1, dtype=torch.float64)
            t0 = time.time()
            for i in range(loops):
                value.add_(param)
            t1 = time.time()
            value.backward()
            t2 = time.time()
            adding[t]   = t1-t0
            backward[t] = t2-t1
            print(f'iter {t}, time add:{adding[t]}, back:{backward[t]}')
            #print(f'\tparam.grad: {param.grad.item()}')
        print(f'Ave Time/Stdv add_:      {np.mean(adding)}, {np.std(adding)}')
        print(f'Ave Time/Stdv backward:  {np.mean(backward)}, {np.std(backward)}\n\n')

        #+=
        adding   = np.zeros(numTrials)
        backward = np.zeros(numTrials)
        for t in range(len(adding)):
            param = torch.ones(1,dtype=torch.float64, requires_grad=True)
            value = torch.ones(1, dtype=torch.float64)
            t0 = time.time()
            for i in range(loops):
                value+=param
            t1 = time.time()
            value.backward()
            t2 = time.time()
            adding[t]   = t1-t0
            backward[t] = t2-t1
            print(f'iter {t}, time add:{adding[t]}, back:{backward[t]}')
        print(f'Ave Time/Stdv +=:        {np.mean(adding)}, {np.std(adding)}')
        print(f'Ave Time/Stdv backward:  {np.mean(backward)}, {np.std(backward)}\n\n')

    if(True): #testing multiply and multiple operations
        print(f'{loops} loops. mul,add vs *,+=')

        #torch.mul
        adding   = np.zeros(numTrials)
        backward = np.zeros(numTrials)
        for t in range(len(adding)):
            param = torch.ones(1,dtype=torch.float64, requires_grad=True)*0.5
            param1= torch.ones(1,dtype=torch.float64, requires_grad=True)*2
            value = torch.ones(1, dtype=torch.float64)
            t0 = time.time()
            for i in range(loops):
                value=torch.add(value,torch.mul(torch.mul(param,param1),1))
            t1 = time.time()
            value.backward()
            t2 = time.time()
            adding[t]   = t1-t0
            backward[t] = t2-t1
            print(f'iter {t}, time ops:{adding[t]}, back:{backward[t]}')
            #print(f'\tparam.grad: {param.grad.item()}')
        print(f'Ave Time/Stdv torch ops: {np.mean(adding)}, {np.std(adding)}')
        print(f'Ave Time/Stdv backward:  {np.mean(backward)}, {np.std(backward)}\n\n')

        #+=,*
        adding   = np.zeros(numTrials)
        backward = np.zeros(numTrials)
        for t in range(len(adding)):
            param = torch.ones(1,dtype=torch.float64, requires_grad=True)*0.5
            param1= torch.ones(1,dtype=torch.float64, requires_grad=True)*2
            value = torch.ones(1, dtype=torch.float64)
            t0 = time.time()
            for i in range(loops):
                value+= param*param1*1
            t1 = time.time()
            value.backward()
            t2 = time.time()
            adding[t]   = t1-t0
            backward[t] = t2-t1
            print(f'iter {t}, time ops:{adding[t]}, back:{backward[t]}')
            #print(f'\tparam.grad: {param.grad.item()}')
        print(f'Ave Time/Stdv torch ops: {np.mean(adding)}, {np.std(adding)}')
        print(f'Ave Time/Stdv backward:  {np.mean(backward)}, {np.std(backward)}\n\n')

#profilingPytorchGradients(3)

#testing CoL logic:

fmWidth,numChannels = 5,3
IT = torch.rand((1,3,fmWidth,fmWidth), dtype=torch.float64)
W = torch.tensor( [ [[1,2,3],[1,2,3],[1,2,3]],\
                    [[4,5,6],[4,5,6],[4,5,6]],\
                    [[7,8,9],[7,8,9],[7,8,9]]], dtype=torch.float64 )
L = torch.ones((5,5), dtype=torch.float64)
bins = np.arange(0,1,1/numBins)
IT_Binned = np.digitize(IT,bins) - 1

#print(f'First channel of IT')
#print(IT[:,0,:,:])
#print(f'First channel of IT_Binned')
#print(IT_Binned[:,0,:,:])
print(f'Spatial Filter: \n{W}')
print(f'Input Tensor: \n{IT}')
print(f'Deep CoOccur: \n{L}')

for chan in range(numChannels):
    for row in range(fmWidth):
        for col in range(fmWidth):
            filteredP = applyFilter(IT,IT_Binned,W,L,(0,chan,row,col))

