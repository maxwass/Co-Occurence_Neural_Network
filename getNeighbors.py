import torch
import time
import numpy as np

#indc is list of tuples
#Input:  list of tuple indeces of dimension N-1, range of values for
#  the new (first) higher dimension
#Output: list of tuple indeces of dimension N, with every combination
#  lower dimensional indeces and value in extraDimRange
def createHigherDimInd(extraDimRange, indc):
    assert len(extraDimRange)>0, 'empty extra dim provided'
    assert len(indc)>0, 'empty set of indeces provided'

    """
    print(extraDimRange)
    for ind in indc:
        print(ind)
    """

    indeces = []
    for c in extraDimRange:
        for tup in indc:
            newInd = (c,) + tup
            indeces.append(newInd)
    return indeces


def borderLocPixel(fmWidth, borderSize, p_index):
    #LEFT, RIGHT, TOP, BOTTOM, NONE

    #p_index = (Batch Size, # channels, Height, Width)
    # columns <==>  width
    # rows    <==>  height
    rowIndx, colIndx = p_index[2], p_index[3]
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

def inChannelNeighbors(fmWidth, spatialFilterWidth, p_index):
    # assuming square feature map Height = Width = fmWidth
    # assuming stride = 1
    # assuming fmWidth > spatialFilterWidth
    assert fmWidth>spatialFilterWidth, \
            f'spat_filt_width {spatialFilterWidth} >= ftre_map_width {fmWidth}'
    assert len(p_index)==4, \
            f'index must be 4 dimensional: (Batch Size, # Channels, \
            Height, Width)'
    assert all(p_index[i]>=0 for i in range(len(p_index))), \
            f'negative coordinates in index: {p_index}'
    borderSize = np.floor_divide(spatialFilterWidth,2)
    assert 2*borderSize<fmWidth, \
            f'spatial filter width too large for feature map width\
            a pixel can be in both LEFT&RIGHT or TOP&BOTTOM'

    """
    print('INPUTS')
    print(f'fmWidth {fmWidth}, spatialFilterWidth {spatialFilterWidth}')
    print(f'p_index {p_index}')
    print(f'borderSize = int_div(spatialFilterWidth,2)=> {borderSize}')
    """

    #p_index = (Batch Size, # channels, Height, Width)
    # columns <==>  width
    # rows    <==>  height
    rowIndx, colIndx = p_index[2], p_index[3]
    assert(0 <= rowIndx <= fmWidth-1) #index cannot be outside of fm
    assert(0 <= colIndx <= fmWidth-1)
    [LEFT,RIGHT,TOP,BOTTOM] = borderLocPixel(fmWidth, borderSize, p_index)

    #We now know if pixel p is on the border, and if so, which part. Use this
    # to construct limits of the row/cols of p's 2D neighborhood
    rowLims, colLims = [], []
    if(LEFT):     colLims = [0,colIndx+borderSize]
    elif(RIGHT):  colLims = [colIndx-borderSize, fmWidth-1]
    else:         colLims = [colIndx-borderSize,colIndx+borderSize]

    if(TOP):      rowLims = [0,rowIndx+borderSize]
    elif(BOTTOM): rowLims = [rowIndx-borderSize, fmWidth-1]
    else:         rowLims = [rowIndx-borderSize,rowIndx+borderSize]

    return rowLims, colLims
