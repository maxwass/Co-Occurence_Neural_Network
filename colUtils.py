import torch
import time
import numpy as np

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

def makeDisjoint(ls,rs,ts,bs,mdl):
    left_top  = set(ls)&set(ts)
    left_bot  = set(ls)&set(bs)
    right_top = set(rs)&set(ts)
    right_bot = set(rs)&set(bs)

    ls_only = set(ls)-left_top-left_bot
    rs_only = set(rs)-right_top-right_bot
    ts_only = set(ts)-left_top-right_top
    bs_only = set(bs)-left_bot-right_bot

    return left_top, left_bot, right_top, right_bot, ls_only, rs_only, ts_only, bs_only

# Takes feature map coordinates of p's neighbors (defined by the spatial
#  filter tensor), and outputs them to p's local coordinates (for
#  indexing into hxwxd spatial filter tensor).
# input:
#   fmIndxs: 2D coordinates in fmap coordinates =>(pRow, pCol) in (0-fmWidth-1,0-fmWidth-1):
# output:
#   list of these fmNeighbors in their respective locations in the 3D sptial fileter
def fm2sf(fmIndxs, sfWidth, pRowCol):
    #Invariant: fmIndxs and sdIndxs have SAME order: ith element of sfIndxs is a
    # transformed coordinate of the ith element of fmIndxsd
    borderSize      = np.floor_divide(sfWidth,2)
    pRow,pCol       = pRowCol[0], pRowCol[1]
    sfIndxs         = []

    for q in fmIndxs:
        sfRow = np.absolute(pRow-q[0]-borderSize)
        sfCol = np.absolute(pCol-q[1]-borderSize)
        sfIndxs.append((sfRow,sfCol)) #appends to END of list - maintaining order
    return sfIndxs

def lims2Coord(rowLims,colLims):
    coords = []
    for row in np.arange(rowLims[0], rowLims[1]+1):
        for col in np.arange(colLims[0], colLims[1]+1):
            coords.append((row,col))
    return coords
def borderLocPixel(fmWidth, borderSize, twoDimIndex):
    #output: LEFT, RIGHT, TOP, BOTTOM booleans

    # columns <==>  width
    # rows    <==>  height
    rowIndx, colIndx = twoDimIndex[0], twoDimIndex[1]
    assert(0 <= rowIndx <= fmWidth-1) #index cannout be outside of fm
    assert(0 <= colIndx <= fmWidth-1)


    # (0,0) is TOP LEFT,
    # (fmWidth-1,fmWidth-1) is BOTTOM RIGHT
    LEFT,RIGHT,TOP,BOTTOM = False,False,False,False
    if(colIndx < borderSize):             LEFT = True
    if(colIndx > (fmWidth-1)-borderSize): RIGHT = True
    if(rowIndx < borderSize):             TOP = True
    if(rowIndx > (fmWidth-1)-borderSize): BOTTOM = True
    loc = [LEFT,RIGHT,TOP,BOTTOM]

    if(False):
        s = f'pixel ({rowIndx}, {colIndx}) -> '
        if(LEFT):   s+= "LEFT "
        if(RIGHT):  s+= "RIGHT "
        if(TOP):    s+= "TOP "
        if(BOTTOM): s+= "BOTTOM "
        if(not any(loc)): s+= "NOT ON BORDER"
        print(s)

    return [LEFT,RIGHT,TOP,BOTTOM]

def inChannelNeighbors(fmWidth, spatialFilterWidth, pRowCol):
    # columns <==>  width
    # rows    <==>  height
    rowIndx, colIndx = pRowCol[0], pRowCol[1]
    assert(0 <= rowIndx <= fmWidth-1), f'fm row: {rowIndx} must be in [0,{fmWidth-1}]'
    assert(0 <= colIndx <= fmWidth-1), f'fm row: {colIndx} must be in [0,{fmWidth-1}]'

    # assuming square feature map Height = Width = fmWidth
    # assuming stride = 1
    # assuming fmWidth > spatialFilterWidth
    assert fmWidth>spatialFilterWidth, \
            f'spat_filt_width {spatialFilterWidth} >= ftre_map_width {fmWidth}'
    assert (0<=rowIndx<=(fmWidth-1)) and (0<=colIndx<=(fmWidth-1)), \
            f'negative coordinates in index: ({rowIndx}, {colIndx})'
    borderSize = np.floor_divide(spatialFilterWidth,2)
    assert 2*borderSize<fmWidth, \
            f'spatial filter width too large for feature map width\
            a pixel can be in both LEFT&RIGHT or TOP&BOTTOM'

    if(False):
        print('INPUTS')
        print(f'fmWidth {fmWidth}, spatialFilterWidth {spatialFilterWidth}')
        print(f'p_index {p_index}')
        print(f'borderSize = int_div(spatialFilterWidth,2)=> {borderSize}')

    # columns <==>  width
    # rows    <==>  height
    [LEFT,RIGHT,TOP,BOTTOM] = borderLocPixel(fmWidth, borderSize, (rowIndx,colIndx))

    #We now know if pixel p is on the border, and if so, which part. Use this
    # to construct limits of the row/cols of p's 2D neighborhood
    rowLims, colLims = [], []
    if(LEFT):     colLims = [0,colIndx+borderSize]
    elif(RIGHT):  colLims = [colIndx-borderSize, fmWidth-1]
    else:         colLims = [colIndx-borderSize,colIndx+borderSize]

    if(TOP):      rowLims = [0,rowIndx+borderSize]
    elif(BOTTOM): rowLims = [rowIndx-borderSize, fmWidth-1]
    else:         rowLims = [rowIndx-borderSize,rowIndx+borderSize]

    return rowLims, colLims # only begin/end index. Must create array to iterate over

# CoL uses channels around channel of interest.
# Ex) if numChannels=5 in the layer, current output channel p_index[1]=3,
#       and spatialFilterDepth = 3 ---> output = channels 2,3,4
# Ex) if numChannels=5 in the layer, current output channel p_index[1]=0,
 #       and spatialFilterDepth = 3 ---> output = channels 4,0,1 (wraps around)
def neighborChannels(numChannels, spatialFilterDepth, outChannel):
    ds = np.floor_divide(spatialFilterDepth,2) #depthSize =~borderSize for depth

    assert spatialFilterDepth<=numChannels, \
            f'depth of spatial filter {spatialFilterDepth} is greater\
            than number of channels {numChannels}'
    firstChan = outChannel-ds
    lastChan  = outChannel+ds
    neighChannels = np.arange(firstChan,lastChan+1,1)
    #print(f'current chan: {outChannel}\nNeighbChan: {neighChannels}')

    if( firstChan < 0 ):
        for i in np.arange(np.absolute(firstChan)):
            neighChannels[i] += numChannels

    elif( lastChan > (numChannels-1) ): #bc numChannels>spatialFilterDepth elif
        indexFirstChanAbove = spatialFilterDepth - (lastChan - (numChannels-1))
        for i in np.arange(indexFirstChanAbove, spatialFilterDepth):
            neighChannels[i] -= numChannels #shift back to beggining channels

    return neighChannels

# OT: normalized input tensor
#
def normalizeTensorPerBatch(IT):

    #calculate the minimum value in each batch
    # add this minumum value to each batch pointwise-> range of values now [0,old_max+old_min] per batch
    numBatches, numChannels,fmHeight,fmWidth = IT.shape
                                                          # (numBatches,,numChannels*fmWidth^2)
    batchVecs            = torch.reshape(IT, (numBatches, numChannels*fmHeight*fmWidth))
    batchMinVals_raw , _ = torch.min(batchVecs,1)          # (numBatches)
    batchMinVals         = -1*batchMinVals_raw # if(minVal<0): add -1*minVal
                                               # if(minVal>0): add -1*minVal
    reshapeBatchMinVals = batchMinVals.view(numBatches,1) # (numBatches,1)
                                                          # broadcasting will make addition
                                                          # (numBatches, max(dim 2 of x,dim 2 of y))

    addMinPerBatch      = torch.add(batchVecs, reshapeBatchMinVals)

    # get max value of all channels in each batch
    #round values to get rid of tiny values?
    # divide by the maximum values of each batch
    newMaxVals, _     = torch.max(addMinPerBatch,1)
    reshapeNewMaxVals = newMaxVals.view(numBatches,1)
    divMaxPerBatch    = torch.div(addMinPerBatch, reshapeNewMaxVals)

    DEBUG=False
    if(DEBUG):
        input('entering per batch broadcasting,..')
        print(f'Input Tensor Shape: \n{IT.shape}\n')
        print(f'Input Tensor: \n{IT}\n')
        print(f'Batch->Vector:\n{batchVecs}\n')
        print(f'(-1) * minimum values/batch:\n{batchMinVals}\n')
        print(f'reshape min values for broadcasting: {reshapeBatchMinVals.shape}\n')
        print(f'add in min values: \n{addMinPerBatch}\n')

        print(f'maximum values/batch:\n{newMaxVals}\n')
        print(f'reshape max values for broadcasting: shape: {reshapeNewMaxVals.shape}\n {reshapeNewMaxVals}')
        print(f'shape of tensor for div: {addMinPerBatch.shape}\n')
        print(f'divide out max values: \n{divMaxPerBatch}\n')
        input('leaving per batch broadcasting,..')

    # now all values in each batch are in range [0,1]
    # bring back to regular shape
    IT_normalized_per_batch = torch.reshape(divMaxPerBatch, (numBatches,numChannels,fmHeight,fmWidth))

    return IT_normalized_per_batch

def normalizeTensorPerChannel(IT):

    #calculate the minimum value in each channel
    # add this minumum value to each channel pointwise-> range of values now [0,old_max+old_min] per batch
    numBatches, numChannels,fmHeight,fmWidth = IT.shape
                                                          # (numBatches*numChannels,fmWidth^2)
    channelVecs            = torch.reshape(IT, (numBatches*numChannels, fmHeight*fmWidth))
    channelMinVals_raw , _ = torch.min(channelVecs,1)          # (numBatches)
    channelMinVals         = -1*channelMinVals_raw # if(minVal<0): add -1*minVal
                                                   # if(minVal>0): add -1*minVal
    reshapeChannelMinVals = channelMinVals.view(numBatches*numChannels,1) # (numBatches*numChannels,1)
                                                          # broadcasting will make addition
                                                          # (numBatches*numChannels, max(dim 2 of x,dim 2 of y))
    addMinPerChannel      = torch.add(channelVecs, reshapeChannelMinVals)

    # get max value of each channels
    #round values to get rid of tiny values?
    # divide by the maximum values of each channel
    newMaxVals, _     = torch.max(addMinPerChannel,1)
    reshapeNewMaxVals = newMaxVals.view(numBatches*numChannels,1)
    divMaxPerChannel    = torch.div(addMinPerChannel, reshapeNewMaxVals)

    DEBUG=False
    if(DEBUG):
        input('entering per channel broadcasting,..')
        print(f'Input Tensor Shape: \n{IT.shape}\n')
        print(f'Input Tensor: \n{IT}\n')
        print(f'Channel->Vector:\n{channelVecs}\n')
        print(f'(-1) * minimum values/channel:\n{channelMinVals}\n')
        print(f'reshape min values for broadcasting: {reshapeChannelMinVals.shape}\n')
        print(f'add in min values: \n{addMinPerChannel}\n')

        print(f'maximum values/Channel:\n{newMaxVals}\n')
        print(f'reshape max values for broadcasting: shape: {reshapeNewMaxVals.shape}\n {reshapeNewMaxVals}')
        print(f'shape of tensor for div: {addMinPerChannel.shape}\n')
        print(f'divide out max values: \n{divMaxPerChannel}\n')
        input('leaving per chan broadcasting,..')

    # now all values in each channel are in range [0,1]
    # bring back to regular shape
    IT_normalized_per_channel = torch.reshape(divMaxPerChannel, (numBatches,numChannels,fmHeight,fmWidth))

    return IT_normalized_per_channel
