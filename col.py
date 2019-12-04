import torch
import time
import numpy as np
from colUtils import *
from lookupTable import *

TIMING = False
def CoL(IT, W, L, batchNormActBounds, indxLookUpTable):
    lowerBound, higherBound = batchNormActBounds[0],batchNormActBounds[1]
    (numBatches, numChannels, fmHeight, fmWidth) = IT.size()
    assert len(IT.size())==4, f'Input Tensor not 4D {IT.size()}'
    assert fmHeight==fmWidth, f'feature map not square: {IT.size()}'
    (k_h,k) = L.size()
    assert len(L.size())==2 and k_h==k, f'deep CoOccur L not square: {L}'
    #quantize inputTensor values
    #each value in this tensor is now the index of that corresponding pixel
    # into L
    #SPEEDUP: use native pytorch quantization technique (github: torch
    # searchsorted), or build own
    #TODO make bins an input to CoL
    (k,k) = L.size()
    bins  = np.linspace(lowerBound, higherBound,k+1,endpoint=True)


    if(TIMING):
        timeBin, timeUtil, timeAFRest = 0.0, 0.0, 0.0
        t0 = time.time()
    IT_Binned = np.digitize(IT,bins) - 1
    if(TIMING):
        t1 = time.time()
        timeBin = t1-t0

    #remove this for time saving -> feed in OT to reuse?
    OT = torch.empty_like(IT, dtype=torch.float32)
    #SPEEDUP OPTIONS: broadcasting, parrallelizing, etc
    for b in range(numBatches):
        for c in range(numChannels):
            for row in range(fmHeight):
                for col in range(fmWidth):
                    p = (b,c,row,col)
                    if(TIMING):
                        OT[b,c,row,col], [tUt,tRes] = applyFilter(IT,IT_Binned,W,L,p,indxLookUpTable)
                        timeUtil  += tUt
                        timeAFRest+=tRes
                    else:
                        OT[b,c,row,col] = applyFilter(IT,IT_Binned,W,L,p,indxLookUpTable)

    if(TIMING):
        times = np.zeros((3,), dtype=np.float32)
        times[0], times[1], times[2] = timeBin, timeUtil, timeAFRest
        return OT, times
    else:
        return OT

#Returns: apply filter: OT[p] = sum over q in N(p): W[q]*L[quant(p),quant(q)]*IC[q]
# IT: Input Tensor: torch tensor (Batch,Channels,Height,Width)
# IT_binned: Numpy ndarray/pytorch tensor with binned activation values.
#             Same size as IT.
# W: spatial filter. pytorch tensor with tracked gradients.
# L: Deep Cooccurence matrix. pytorch tensor with tracked gradients.
# p_4d: 4d index of current point of interest
# indxLookUpTable: large lookup table storing the lists of fm,sf neighbors
#              given each pixel p (4D)
def applyFilter(IT,IT_binned,W,L,p_4D, indxLookUpTable):
    (numBatch,numChannels,fmHeight,fmWidth)  = IT.size()
    (sfHeight,sfWidth,sfDepth) = W.size()
    pBatch, pChan, pRow, pCol = p_4D[0], p_4D[1], p_4D[2], p_4D[3]

    if(TIMING):
        t0 = time.time()

    #sfNeighbors is a transformed version of fmNeighbors and maintains the same order of indx's
    numNeighbors = indxLookUpTable[pBatch,pChan,pRow,pCol]['numNeighbors']
    fmNeighbors  = indxLookUpTable[pBatch,pChan,pRow,pCol]['fmNeighbors']
    sfNeighbors  = indxLookUpTable[pBatch,pChan,pRow,pCol]['sfNeighbors']

    if(TIMING):
        t1 = time.time()
        timeUtils = t1-t0

    assert len(fmNeighbors)==len(sfNeighbors),\
            f'fmNeighbors and sfNeighbors should contain the same \
            neighbors in transformed coordinates'

    #fill a tensor of same size as spatial filter with:
    # sf_cons - spatial coefficients used at their respective
    #           locations. 0's if unused.
    # ne_cons - neighbors of the current pixel. 0's if outside of
    #           kernel (if pixel is on a boundary)
    DEBUG = False
    if(DEBUG):
        sf_cons        = torch.zeros_like(W,dtype=torch.float32)
        neighbors_cons = torch.zeros_like(W,dtype=torch.float32)
        L_cons         = torch.zeros_like(W,dtype=torch.float32)

    #to speed up: https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/4
    torchAccOp = False
    if(torchAccOp):
        filteredP_temp = torch.zeros(numNeighbors, dtype=torch.float32)
    else:
        filteredP = torch.zeros(1, dtype=torch.float32)

    for i in range(numNeighbors):
        q_fm, q_sf = fmNeighbors[i], sfNeighbors[i]
        #print(f'p_4d: {p_4D}, q_fm: {q_fm}, q_sf: {q_sf}')
        pBin, qBin = IT_binned[p_4D], IT_binned[q_fm]
        w_q  = W[q_sf]
        l_pq = L[pBin,qBin]
        I_q  = IT[q_fm]
        if(torchAccOp):
                filteredP_temp[i] = w_q*l_pq*I_q
        else:
                filteredP += w_q*l_pq*I_q
        if(DEBUG):
            neighbors_cons[q_sf] = I_q
            sf_cons[q_sf]        = w_q
            L_cons[q_sf]         = l_pq

    if(TIMING):
        t2 = time.time()
        timeApplyFilterRest = t2-t1

    if(torchAccOp):
        if(TIMING):
            return torch.sum(filteredP_temp), [timeUtils, timeApplyFilterRest]
        else:
            return torch.sum(filteredP_temp)
    else:
        if(TIMING):
            return filteredP, [timeUtils, timeApplyFilterRest]
        else:
            return filteredP

