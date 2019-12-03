import torch
import time
import numpy as np

from colUtils import *

def test_borderLocPixel():
    fmWidth, sfWidth, sfDepth  = 8,5,3
    borderSize = np.floor_divide(sfWidth,2)

    ls,rs,ts,bs,mdl = genBorderIndeces(fmWidth, borderSize)
    ## test borderLocPixel
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

    #test inChannelNeighbors
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

def test_neighborChannels():
    #test neighborChannels
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

def test_sfNeighbors():
    fmWidth,numChannels,sfDims = 8,5,(3,3,3) #Height, Width, Depth
    borderSize = np.floor_divide(3,2)
    ls,rs,ts,bs,mdl = genBorderIndeces(fmWidth, borderSize)
    lt, lb, rt, rt, l, r, t, b = makeDisjoint(ls,rs,ts,bs,mdl)

    # 3 cases:
    #(1) fmWidth=5, sfWidth=3
    #(2) fmWidth=8, sfWidth=3
    #(3) fmWidth=8, sfWidth=5

    a = []
    fmWidth, sfWidth = 5 , 3

    middleSFNeighbors = []
    for i in range(sfWidth):
        for j in range(sfWidth):
            middleSFNeighbors.append((i,j))

    #(point, true sf neighbor indexes)
    #(1) fmWidth=5, sfWidth=3
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
    #(2) fmWidth=8, sfWidth=3
    fmWidth, sfWidth = 8 , 3
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

    #(3) fmWidth=8, sfWidth=5
    fmWidth, sfWidth = 8 , 5
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
            #TODO: create new variables in loop:
            # https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-       through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-  true-when-calling-backward-the-first-time/6795/4
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



def brute_force_normalize(IT):
    numBatches, numChannels,fmHeight,fmWidth = IT.shape
    OT_batch_norm, OT_batch_norm_min_add = torch.zeros_like(IT),torch.zeros_like(IT)
    OT_chann_norm, OT_chann_norm_min_add = torch.zeros_like(IT), torch.zeros_like(IT)

    DEBUG=False
    #find min of each batch and channel
    minEachChannel = np.zeros((numBatches,numChannels))
    for b in range(numBatches):
        for chan in range(numChannels):
            currMin = np.inf
            for row in range(fmHeight):
                for col in range(fmWidth):
                    val = IT[b,chan,row,col]
                    if (val<currMin):
                        currMin=val
            minEachChannel[b,chan] = currMin

    minEachBatch = np.min(minEachChannel,axis=1)

    #add in minumum
    for b in range(numBatches):
        for chan in range(numChannels):
            for row in range(fmHeight):
                for col in range(fmWidth):
                    OT_batch_norm_min_add[b,chan,row,col] = IT[b,chan,row,col] + (-1)*minEachBatch[b]
                    OT_chann_norm_min_add[b,chan,row,col] = IT[b,chan,row,col] + (-1)*minEachChannel[b,chan]

    #find max of each batch/channel
    maxEachChannel = np.zeros((numBatches,numChannels))
    maxEachBatch = np.zeros(numBatches)
    for b in range(numBatches):
        currMaxBatch  = -np.inf
        for chan in range(numChannels):
            currMaxChan = -np.inf
            for row in range(fmHeight):
                for col in range(fmWidth):
                    val_chan  = OT_chann_norm_min_add[b,chan,row,col]
                    val_batch = OT_batch_norm_min_add[b,chan,row,col]
                    if (val_chan>currMaxChan):
                        currMaxChan = val_chan
                    if (val_batch>currMaxBatch):
                        currMaxBatch = val_batch
            maxEachChannel[b,chan] = currMaxChan
        maxEachBatch[b] = currMaxBatch

   #divide by maximum
    for b in range(numBatches):
        for chan in range(numChannels):
            for row in range(fmHeight):
                for col in range(fmWidth):
                    OT_batch_norm[b,chan,row,col] = OT_batch_norm_min_add[b,chan,row,col]/maxEachBatch[b]
                    OT_chann_norm[b,chan,row,col] = OT_chann_norm_min_add[b,chan,row,col]/maxEachChannel[b,chan]

    if(DEBUG):
        print(f'IT dims: ({numBatches}, {numChannels},{fmWidth},{fmWidth})')
        print(f'IT:\n{IT}')
        print(f'minimum of each channel:\n{minEachChannel}')
        print(f'minimum of each batch:\n{minEachBatch}')
        print(f'Batch   Norm Added min:\n{OT_batch_norm_min_add}\n')
        print(f'Channel Norm Added min:\n{OT_chann_norm_min_add}\n')
        print(f'max of each channel:\n{maxEachChannel}')
        print(f'max of each batch:\n{maxEachBatch}')
        print(f'Batch   Norm Div by max:\n{OT_batch_norm}\n')
        print(f'Channel Norm Div by max:\n{OT_chann_norm}\n')
        input('...')

    return OT_batch_norm, OT_chann_norm


def compareBrute2Broadcast(numBatches,numChannels,fmWidth):
    IT = torch.randn((numBatches,numChannels,fmWidth,fmWidth), dtype=torch.float32)

    t0 = time.time()
    OT_batch_normed_brute, OT_channel_normed_brute = brute_force_normalize(IT)

    t1 = time.time()
    IT_batch_normed = normalizeTensorPerBatch(IT)
    IT_channel_normed = normalizeTensorPerChannel(IT)
    t2 = time.time()
    brute_time, broad_time = (t1-t0), (t2-t1)

    print(f'IT dims: ({numBatches}, {numChannels},{fmWidth},{fmWidth})')
    print(f'\tbrute: {brute_time}, broad: {broad_time}')
    assert torch.allclose(OT_batch_normed_brute,IT_batch_normed),\
        f'Broadcasting in batch normalization not working as expected\
        \nbrute force:\n{OT_batch_normed_brute}\
        \nbroadcast:  \n{IT_batch_normed}'
    assert torch.allclose(OT_channel_normed_brute,IT_channel_normed),\
        f'Broadcasting in channel normalization not working as expected'

    return [brute_time, broad_time]

def test_normalizeTensor(highBatch,highChan,highFm):
    #try different number of Batches/channels/fm sizes
    for numBatches in np.arange(1,highBatch,1):
        for numChannels in np.arange(1,highChan,3):
            for fmWidth in np.arange(2,highFm,10):
                compareBrute2Broadcast(numBatches,numChannels,fmWidth)

test_normalizeTensor(2,8,42)
test_borderLocPixel()
test_neighborChannels()
test_sfNeighbors()
#profilingPytorchGradients(3)
