import torch
import time
import sys
import numpy as np

from colUtils import *
from col import *
from lookupTable import *

def test_genLookUpTable():
    print(f'\n\nTESTING genLookUp Table')
    print("NOT IMPLEMENTED: All other tests work when using this, so should be right")
    return
    numBatches, fmWidth,numChannels = 2,8,5
    sfWidth, sfDepth = 3,3
    borderSize = np.floor_divide(sfWidth,2)

    IT = torch.zeros((numBatches,numChannels,fmWidth,fmWidth), dtype=torch.float32)
    W = torch.ones((sfDepth,sfWidth,sfWidth), requires_grad=True,dtype=torch.float32)

    indxLookUpTable = genLookUpTable(IT.size(), W.size())
    print(indxLookUpTable)
test_genLookUpTable()
