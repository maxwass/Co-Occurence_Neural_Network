import torch
import time
import numpy as np

chan_row_col_dtype = np.dtype([('channel', np.int16), ('row', np.int16),('col', np.int16)])


def CoL(IT, W, L):
    (numBatches, numChannels, fmHeight, fmWidth) = inputTensor.size()
    assert len(IT.size())==4, f'Input Tensor not 4D {IT.size()}'
    assert len(p)==4, f'expecting coordinate p {p_4D} to be 4D'
    assert fmHeight==fmWidth, f'feature map not square: {IT.size()}'


    #quantize inputTensor values
    #each value in this tensor is now the index of that corresponding pixel
    # into L
    #SPEEDUP: use native pytorch quantization technique (github: torch
    # searchsorted), or build own
    bins = np.arange(0,1,1/numBins)
    inputTensorBinned = np.digitize(inputTensor,bins) - 1

    (numBatches,numChannels,height,width) = inputTensor.size()

    #NEED TO BE TRACKED??
    OT = torch.empty_like(inputTensor)
    #SPEEDUP OPTIONS: broadcasting, parrallelizing, etc
    for b in range(numBatches):
        for c in range(numChannels):
            for row in range(height):
                for col in range(width):
                    OT[b,c,row,col] = applyFilter(inputTensor,inputTensorBinned,W,L,(b,c,row,col))


    return OT
#apply filter: IT[p] = sum over q in N(p): W[q]*L[quant(p),quant(q)]*IC[q]
# IT: Input Tensor: torch tensor (Batch,Channels,Height,Width)
# IT_binned: Numpy ndarray/pytorch tensor with binned activation values.
#             Same size as IT.
# W: spatial filter. pytorch tensor with tracked gradients.
# L: Deep Cooccurence matrix. pytorch tensor with tracked gradients.
# p_4d: 4d index of current point of interest.
def applyFilter(IT,IT_binned,W,L,p_4D):
    (numBatch,numChannels,fmHeight,fmWidth)  = IT.size()
    (sfHeight,sfWidth,sfDepth) = W.size()
    b, pChan, pRow, pCol       = p_4D[0], p_4D[1], p_4D[2], p_4D[3]

    fmRowLims, fmColLims = inChannelNeighbors(fmWidth, sfWidth, (pRow,pCol))
    fmNeighbors          = lims2Coord(fmRowLims,fmColLims)
    sfNeighbors          = fm2sf(fmNeighbors, sfWidth, (pRow,pCol))
    nChannels            = neighborChannels(numChannels, sfDepth, pChan)

    assert len(fmNeighbors)==len(sfNeighbors),\
            f'fmNeighbors and sfNeighbors should contain the same \
            neighbors in transformed coordinates'

    filteredP = torch.zeros(1, dtype=torch.float64)

    #neighborChannels is a list of channels from which to pull p's neighbors from.
    # The values in this list are indeces of the corresponding channels in order of
    #  their use (from back to front).
    #When indexing into the spatial filter W, we only care about the relative location
    #  to p, so wChan is the depth index into W. We start at the back of W and move forward.
    #sfNeighbors is a transformed version of fmNeighbors and maintains the same order of indx's

    #fill a tensor of same size as spatial filter with:
    # sf_cons - spatial coefficients used at their respective
    #           locations. 0's if unused.
    # ne_cons - neighbors of the current pixel. 0's if outside of
    #           kernel (if pixel is on a boundary)
    DEBUG = True
    if(DEBUG):
        sf_cons        = torch.zeros_like(W,dtype=torch.float64)
        neighbors_cons = torch.zeros_like(W,dtype=torch.float64)
        L_cons         = torch.zeros_like(W,dtype=torch.float64)
    #neighbor channels nChannels are in order from first to last
    for chanIndexSf, chanIndexFm in enumerate(nChannels):
        for i in range(len(fmNeighbors)):
            q_fm, q_sf = fmNeighbors[i], sfNeighbors[i]

            #3D index. Ensure this indexing is correct. W is 3D
            q_sf = (chanIndexSf,)   + q_sf   #3D
            q_fm = (b,chanIndexFm,) + q_fm   #4D

            w_q  = W[q_sf]

            l_pq = L[IT_binned[p_4D]][IT_binned[q_fm]] #check this type
            I_q  = IT[q_fm]
            #print(f'\nq_fm: {q_fm}, IT[q]= {I_q}')
            #print(f'q_sf: {q_sf}, weight_q: {w_q}')
            #input('...')
            #print(f'IT values: ({IT[p_4D]}, {IT[q_fm]})')
            #print(f'L indeces: ({IT_binned[p_4D]}, {IT_binned[q_fm]})')
            filteredP += w_q*l_pq*I_q
            if(DEBUG):
                neighbors_cons[q_sf] = I_q
                sf_cons[q_sf]        = w_q
                L_cons[q_sf]         = l_pq

    #input(f'Neighbors around (channel,row,col) = (({pChan},{pRow},{pCol})')
    #print(f'IT[p] = {IT[p_4D]}')
    #print(f'Neighbors Used: \n{neighbors_cons}')
    #print(f'W weights used: \n{sf_cons}')
    #print(f'L used: {L_cons}')
    #input('...')
    return filteredP

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
