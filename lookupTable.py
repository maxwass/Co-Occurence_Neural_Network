import torch
import time
import numpy as np
from colUtils import *

#define necessary data types

#maxNumNeighbors = 27
a = np.int16(1)
dt_4D = type((a,a,a,a))#np.dtype(('uint8', (4,)))
dt_3D = type((a,a,a,a))#np.dtype(('uint8', (3,)))
#dt = np.dtype([('numNeighbors', np.uint16),\
#     ('fmNeighbors', dt_4D, maxNumNeighbors),\
#     ('sfNeighbors', dt_3D, maxNumNeighbors)])

#p_4d = np.zeros(maxNumNeighbors,dtype=dt_4D)
#p_3d = np.ones(maxNumNeighbors,dtype=dt_3D)

def genLookUpTable(IT_dims, W_dims):
    maxNumNeighbors = np.prod(np.asarray(W_dims))
    dt = np.dtype([('numNeighbors', np.uint16),\
            ('fmNeighbors', dt_4D, maxNumNeighbors),\
            ('sfNeighbors', dt_3D, maxNumNeighbors)])

    numBatches, numChannels, fmHeight, fmWidth = IT_dims[0], IT_dims[1], IT_dims[2], IT_dims[3]
    sfDepth, sfHeight, sfWidth = W_dims[0], W_dims[1],  W_dims[2]
    assert fmHeight==fmWidth, f'fmHeight {fmHeight} must equal fmWidth {fmWidth}'
    assert sfHeight==sfWidth, f'sfHeight {sfHeight} must equal sfWidth {sfWidth}'

    lookupTableIndxs = np.zeros((numBatches, numChannels, fmWidth, fmWidth), dtype=dt)
    DEBUG = False
    for pBatch in range(numBatches):
        for pChan in range(numChannels):
            for pRow in range(fmHeight):
                for pCol in range(fmWidth):
                    if(DEBUG):
                        print(f'p: ({pBatch},{pChan},{pRow},{pCol})')
                        print(lookupTableIndxs[pBatch,pChan,pRow,pCol]['numNeighbors'])
                        print(lookupTableIndxs[pBatch,pChan,pRow,pCol]['fmNeighbors'])
                        print(lookupTableIndxs[pBatch,pChan,pRow,pCol]['sfNeighbors'])
                        input('...')
                    fmRowLims, fmColLims = inChannelNeighbors(fmWidth, sfWidth, (pRow,pCol))
                    fmNeighbors          = lims2Coord(fmRowLims,fmColLims)
                    sfNeighbors          = fm2sf(fmNeighbors, sfWidth, (pRow,pCol))
                    nChannels            = neighborChannels(numChannels, sfDepth, pChan)

                    fmNeighbors_ = np.empty(maxNumNeighbors, dtype=dt_4D)
                    sfNeighbors_ = np.empty(maxNumNeighbors, dtype=dt_3D)

                    numNeighbors = len(fmNeighbors)*len(nChannels)
                    lookupTableIndxs[pBatch,pChan,pRow,pCol]['numNeighbors'] = numNeighbors
                    j=0
                    for chanIndexSf, chanIndexFm in enumerate(nChannels):
                        for i in range(len(fmNeighbors)):
                            q_fm, q_sf = fmNeighbors[i], sfNeighbors[i]
                            q_sf = (chanIndexSf,) + q_sf
                            q_fm = (pBatch, chanIndexFm,) + q_fm
                            #insert into lookupTableIndxs
                            lookupTableIndxs[pBatch,pChan,pRow,pCol]['fmNeighbors'][j] = q_fm
                            lookupTableIndxs[pBatch,pChan,pRow,pCol]['sfNeighbors'][j] = q_sf
                            if(False):
                                print(f'insert {j}th Neighbor\n\tq_fm: {q_fm}\n\tq_sf: {q_sf}')
                                print(f'number of Neighbors:')
                                print(lookupTableIndxs[pBatch,pChan,pRow,pCol]['numNeighbors'])
                                print(f'fmNeighbors:')
                                print(lookupTableIndxs[pBatch,pChan,pRow,pCol]['fmNeighbors'])
                                print(f'sfNeighbors:')
                                print(lookupTableIndxs[pBatch,pChan,pRow,pCol]['sfNeighbors'])
                                input('...')
                            j+=1

    return lookupTableIndxs
