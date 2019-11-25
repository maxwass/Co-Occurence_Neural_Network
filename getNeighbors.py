import torch
import time
import numpy as np

chan_row_col_dtype = np.dtype([('channel', np.int16), ('row', np.int16),('col', np.int16)])
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


def getNeighborhoodIndeces(fmWidth, numChannels, sfDims, p):

    sfWidth, sfDepth = sfDims[1],sfDims[2] #Height, Width, Depth
    bS = np.floor_divide(sfWidth,2)
    pChan,pRow,pCol = p[1], p[2],p[3]
    rows, cols = inChannelNeighbors(fmWidth, sfWidth, p)
    nC         = neighborChannels(numChannels, sfDepth, p)

    print(f'chan,(row,col): {p[1]}, ({p[2]},{p[3]})')
    print(f'rowsLims: {rows}')
    print(f'colLims:  {cols}')
    print(f'Channels: {nC}')
    #create 3D indeces
    #TO DO:
    #predefine array to make faster: compare this to simple array append
    #map this onto 3D tensor?
    neighbors = np.zeros(len(rows)*len(cols)*len(nC),dtype=chan_row_col_dtype)#('int16,int16,int16'))
    spatialWieghtIndxNeighbors=np.zeros(len(rows)*len(cols)*len(nC),dtype=chan_row_col_dtype)
    neighbors.fill(np.nan) #will crash in indexing if we dont fill all of them
    spatialWieghtIndxNeighbors.fill(np.nan) #will crash in indexing if we dont fill all of them

    i = 0
    for indxChan, chan in enumerate(nC):
        for row in rows:
            for col in cols:
                neighbors[i] = (chan,row,col) #this is likely cast to dtype^ Access unclear
                spatialWieghtIndxNeighbors[i] = (indxChan,pRow-row-bS,pCol-col-bS)#TODO
                i+=1

    return spatialWieghtIndxNeighbors, neighbors

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

    if(False):
        print('INPUTS')
        print(f'fmWidth {fmWidth}, spatialFilterWidth {spatialFilterWidth}')
        print(f'p_index {p_index}')
        print(f'borderSize = int_div(spatialFilterWidth,2)=> {borderSize}')

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

# CoL uses channels around channel of interest.
# Ex) if numChannels=5 in the layer, current output channel p_index[1]=3,
#       and spatialFilterDepth = 3 ---> output = channels 2,3,4
# Ex) if numChannels=5 in the layer, current output channel p_index[1]=0,
 #       and spatialFilterDepth = 3 ---> output = channels 4,0,1 (wraps around)
def neighborChannels(numChannels, spatialFilterDepth, p_index):
    ds = np.floor_divide(spatialFilterDepth,2) # borderSize for depth
    outChannel = p_index[1]

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
