import torch
import time
import numpy as np

chan_row_col_dtype = np.dtype([('channel', np.int16), ('row', np.int16),('col', np.int16)])


def CoL(inputTensor, W, L):

    #quantize inputTensor values
    #each value in this tensor is now the index of that corresponding pixel
    # into L
    #SPEEDUP: use native pytorch quantization technique, or build own
    bins = np.arange(0,1,1/numBins)
    inputTensor_quant = np.digitize(t,bins) - 1

    (numBatches,numChannels,height,width) = inputTensor.size()

    #SPEEDUP OPTIONS: broadcasting, parrallelizing, etc
    for b in range(numBatches):
        for c in range(numChannels):
            for row in range(height):
                for col in range(width):
                    OC[b,c,row,col] = applyFilter(inputTensor,W,L,(b,c,row,col))

#apply filter: OC[p] = sum over q in N(p): W[q]*L[quant(p),quant(q)]*IC[q]
def applyFilter(IC,W,L,p):
    print("Apply Filter not implemented")




# Takes feature map coordinates of p's neighbors (defined by the spatial
#  filter tensor), and outputs them to p's local coordinates (for
#  indexing into hxwxd spatial filter tensor).
# input:
#   fmNeighbors: 2D coordinates in fmap coordinates =>(pRow, pCol) in (0-fmWidth-1,0-fmWidth-1):
# output:
#   list of these fmNeighbors in their respective locations in the 3D sptial fileter
def fm2sf(fmNeighbors, sfWidth, p):

    borderSize      = np.floor_divide(sfWidth,2)
    pRow,pCol       = p[0], p[1]
    sfIndxs         = []

    for q in fmNeighbors:
        sfRow = np.absolute(pRow-q[0]-borderSize)
        sfCol = np.absolute(pCol-q[1]-borderSize)
        sfIndxs.append((sfRow,sfCol))
    return sfIndxs

def lims2Coord(rowLims,colLims):
    coords = []
    for row in np.arange(rowLims[0], rowLims[1]+1):
        for col in np.arange(colLims[0], colLims[1]+1):
            coords.append((row,col))
    return coords
def localNeighb2TensorMask(localNeighbors, sfDims):
    sfMask = torch.new_full((3,3,3), 0, dtype=torch.uint8, requires_grad=False)
    print(sfMask)
    #loop over each index in local neighbors and set its corresponding
    # index in the 0's tensor to 1
    #TODO
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

def inChannelNeighbors(fmWidth, spatialFilterWidth, twoDimIndex):
    # columns <==>  width
    # rows    <==>  height
    rowIndx, colIndx = twoDimIndex[0], twoDimIndex[1]
    assert(0 <= rowIndx <= fmWidth-1) #index cannout be outside of fm
    assert(0 <= colIndx <= fmWidth-1)

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
