import torch
import numpy as np

def getNeighbors(fmWidth, spatialFilterWidth, p_index):
    # assuming square feature map Height = Width = fmWidth
    # assuming stride = 1
    # assuming fmWidth > spatialFilterWidth

    f'fmWidth {fmWidth}, spatialFilterWidth {spatialFilterWidth}'
    f'p_index {p_index}'
    borderSize = np.floor_divide(spatialFilterWidth,2)
    f'borderSize {borderSize}'

    #p_index = (Batch Size, # channels, Height, Width)
    #columns correspond to width
    #rows correspond to height
    rowIndx, colIndx = p_index[2], p_index[3]
    assert(0 <= rowIndx <= fmWidth-1) #index cannout be outside of fm
    assert(0 <= colIndx <= fmWidth-1)

    # (0,0) is TOP LEFT,
    # (fmWidth-1,fmWidth-1) is BOTTOM RIGHT
    LEFT,RIGHT,TOP,BOTTOM = False,False,False,False
    LEFT =True if(colIndx < borderSize) else False
    RIGHT=True if(colIndx > (fmWidth-1)-borderSize) else False
    TOP  =True if(rowIndx < borderSize) else False
    BOTTOM=True if(rowIndx > (fmWidth-1)-borderSize) else False

    rows, cols = {}, {}

    if(LEFT): cols = ...
    elif(RIGHT):
    else:

    if(TOP):
    elif(BOTTOM):
    else:
