import torch
import time
import sys
import numpy as np

from colUtils import *
from col import *

#test applyFilter:
def test_applyFilter():
    print(f'\n\nTESTING applyFilter')
    #test 1: W,L all ones, IT constant per channel

    fmWidth,sfWidth,numChannels,k= 5,3,3,5
    borderSize = np.floor_divide(sfWidth,2)

    IT = torch.zeros((1,3,5,5), dtype=torch.float32)
    IT[:,0,:,:], IT[:,1,:,:], IT[:,2,:,:]  = 1.0, 2.0, 3.0
    W = torch.ones((3,3,3), requires_grad=True,dtype=torch.float32)
    L = torch.ones((k,k), requires_grad=True,dtype=torch.float32)

    bins  = np.linspace(0.0, 3.0,k,endpoint=True)
    IT_Binned = np.digitize(IT,bins) - 1
    #print(bins)
    #print(IT_Binned)

    #print(f'Spatial Filter: \n{W}')
    #print(f'Input Tensor: \n{IT}')
    #print(f'Deep CoOccur: \n{L}')
    #print(f'Input Ten Bin:\n{IT_Binned}')
    ls,rs,ts,bs,mdl            = genBorderIndeces(fmWidth, borderSize)
    lt, lb, rt, rb, l, r, t, b = makeDisjoint(ls,rs,ts,bs,mdl)

    #MIDDLE CHANNEL
    #Only do 1st (middle) Channel atm

    #define derivs
    left_deriv = torch.ones((3,3,3), dtype=torch.float32)
    left_deriv[1,:,:], left_deriv[2,:,:] = 2, 3
    left_deriv[:,:,0]=0.0

    right_deriv = torch.ones((3,3,3), dtype=torch.float32)
    right_deriv[1,:,:], right_deriv[2,:,:] = 2,3
    right_deriv[:,:,2]=0.0

    top_deriv = torch.ones((3,3,3), dtype=torch.float32)
    top_deriv[1,:,:], top_deriv[2,:,:] = 2, 3
    top_deriv[:,0,:]=0.0

    bottom_deriv = torch.ones((3,3,3), dtype=torch.float32)
    bottom_deriv[1,:,:], bottom_deriv[2,:,:] = 2, 3
    bottom_deriv[:,2,:]=0.0

    L_side_deriv = torch.zeros((k,k),dtype=torch.float32)
    L_side_deriv[2,1], L_side_deriv[2,2],  L_side_deriv[2,4] = 6*1,6*2,6*3


    top_left_deriv = torch.ones((3,3,3), dtype=torch.float32)
    top_left_deriv[1,:,:], top_left_deriv[2,:,:] = 2, 3
    top_left_deriv[:,:,0], top_left_deriv[:,0,:] = 0.0, 0.0

    top_right_deriv = torch.ones((3,3,3), dtype=torch.float32)
    top_right_deriv[1,:,:], top_right_deriv[2,:,:] = 2, 3
    top_right_deriv[:,:,2], top_right_deriv[:,0,:] = 0.0, 0.0

    bottom_left_deriv = torch.ones((3,3,3), dtype=torch.float32)
    bottom_left_deriv[1,:,:], bottom_left_deriv[2,:,:] = 2, 3
    bottom_left_deriv[:,:,0], bottom_left_deriv[:,2,:] = 0.0, 0.0

    bottom_right_deriv = torch.ones((3,3,3), dtype=torch.float32)
    bottom_right_deriv[1,:,:], bottom_right_deriv[2,:,:] = 2, 3
    bottom_right_deriv[:,2,:], bottom_right_deriv[:,:,2] = 0.0, 0.0

    L_corner_deriv = torch.zeros((k,k),dtype=torch.float32)
    L_corner_deriv[2,1], L_corner_deriv[2,2],  L_corner_deriv[2,4] = 4*1,4*2,4*3


    mdl_deriv = torch.ones((3,3,3), dtype=torch.float32)
    mdl_deriv[1,:,:], mdl_deriv[2,:,:] = 2, 3

    L_mdl_deriv = torch.zeros((k,k), dtype=torch.float32)
    L_mdl_deriv[2,1], L_mdl_deriv[2,2],  L_mdl_deriv[2,4] = 9*1,9*2,9*3

    #generate look up table for neighbors
    maxNumNeighbors = np.prod(W.size())
    input(f'This should be 27: {maxNumNeighbors}...')
    indxLookUpTable = genLookUpTable(IT.size(), W.size(), maxNumNeighbors)
    print(f'Testing W/L gradients:fmWidth: {fmWidth}, sfWidth: {sfWidth}')
    chan = 2-1
    for p in l:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, left_deriv), \
                f'gradient incorrect for left pixels {p}\nis:{W.grad}\nshould be\n{left_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for left pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in r:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, right_deriv), \
                f'gradient incorrect for right pixels {p}\nis:{W.grad}\nshould be\n{right_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for right pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in t:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, top_deriv), \
                f'gradient incorrect for top pixels {p}\nis:{W.grad}\nshould be\n{top_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for right pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in b:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_deriv), \
                f'gradient incorrect for bottom pixels {p}\nis:{W.grad}\nshould be\n{bottom_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for bottom pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in lt:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, top_left_deriv), \
                f'gradient incorrect for top_left pixels {p}\nis:{W.grad}\nshould be\n{top_left_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for top_left pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in lb:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_left_deriv), \
                f'gradient incorrect for bottom_left pixels {p}\nis:{W.grad}\nshould be\n{bottom_left_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for bottom_left pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in rt:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, top_right_deriv), \
                f'gradient incorrect for top_right pixels {p}\nis:{W.grad}\nshould be\n{top_right_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for top_right pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()


    for p in rb:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_right_deriv), \
                f'gradient incorrect for bottom_right pixels {p}\nis:{W.grad}\nshould be\n{bottom_right_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for bottom_right pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in mdl:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, mdl_deriv), \
                f'gradient incorrect for middle pixels {p}\nis:{W.grad}\nshould be\n{mdl_deriv}'
        assert torch.allclose(L.grad, L_mdl_deriv), \
                f'L gradient incorrect for middle pixels {p}\nis:{L.grad}\nshould be\n{L_mdl_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    #FIRST CHANNEL
    #0th (first) channel
    chan=0
    #define derivs
    left_deriv = torch.ones((3,3,3), dtype=torch.float32)
    left_deriv[2,:,:], left_deriv[0,:,:] = 2, 3
    left_deriv[:,:,0]=0.0

    right_deriv = torch.ones((3,3,3), dtype=torch.float32)
    right_deriv[2,:,:], right_deriv[0,:,:] = 2,3
    right_deriv[:,:,2]=0.0

    top_deriv = torch.ones((3,3,3), dtype=torch.float32)
    top_deriv[2,:,:], top_deriv[0,:,:] = 2, 3
    top_deriv[:,0,:]=0.0

    bottom_deriv = torch.ones((3,3,3), dtype=torch.float32)
    bottom_deriv[2,:,:], bottom_deriv[0,:,:] = 2, 3
    bottom_deriv[:,2,:]=0.0

    L_side_deriv = torch.zeros((k,k), dtype=torch.float32)
    L_side_deriv[1,1], L_side_deriv[1,2],  L_side_deriv[1,4] = 6*1,6*2,6*3


    top_left_deriv = torch.ones((3,3,3), dtype=torch.float32)
    top_left_deriv[2,:,:], top_left_deriv[0,:,:] = 2, 3
    top_left_deriv[:,:,0], top_left_deriv[:,0,:] = 0.0, 0.0

    top_right_deriv = torch.ones((3,3,3), dtype=torch.float32)
    top_right_deriv[2,:,:], top_right_deriv[0,:,:] = 2, 3
    top_right_deriv[:,:,2], top_right_deriv[:,0,:] = 0.0, 0.0

    bottom_left_deriv = torch.ones((3,3,3), dtype=torch.float32)
    bottom_left_deriv[2,:,:], bottom_left_deriv[0,:,:] = 2, 3
    bottom_left_deriv[:,:,0], bottom_left_deriv[:,2,:] = 0.0, 0.0

    bottom_right_deriv = torch.ones((3,3,3), dtype=torch.float32)
    bottom_right_deriv[2,:,:], bottom_right_deriv[0,:,:] = 2, 3
    bottom_right_deriv[:,2,:], bottom_right_deriv[:,:,2] = 0.0, 0.0

    L_corner_deriv = torch.zeros((k,k), dtype=torch.float32)
    L_corner_deriv[1,1], L_corner_deriv[1,2],  L_corner_deriv[1,4] = 4*1,4*2,4*3


    mdl_deriv = torch.ones((3,3,3), dtype=torch.float32)
    mdl_deriv[2,:,:], mdl_deriv[0,:,:] = 2, 3

    L_mdl_deriv = torch.zeros((k,k), dtype=torch.float32)
    L_mdl_deriv[1,1], L_mdl_deriv[1,2],  L_mdl_deriv[1,4] = 9*1,9*2,9*3
    for p in l:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, left_deriv), \
                f'gradient incorrect for left pixels {p}\nis:{W.grad}\nshould be\n{left_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for left pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in r:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, right_deriv), \
                f'gradient incorrect for right pixels {p}\nis:{W.grad}\nshould be\n{right_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for right pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in t:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, top_deriv), \
                f'gradient incorrect for top pixels {p}\nis:{W.grad}\nshould be\n{top_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for right pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in b:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_deriv), \
                f'gradient incorrect for bottom pixels {p}\nis:{W.grad}\nshould be\n{bottom_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for bottom pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in lt:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, top_left_deriv), \
                f'gradient incorrect for top_left pixels {p}\nis:{W.grad}\nshould be\n{top_left_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for top_left pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in lb:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_left_deriv), \
                f'gradient incorrect for bottom_left pixels {p}\nis:{W.grad}\nshould be\n{bottom_left_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for bottom_left pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in rt:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, top_right_deriv), \
                f'gradient incorrect for top_right pixels {p}\nis:{W.grad}\nshould be\n{top_right_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for top_right pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()


    for p in rb:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_right_deriv), \
                f'gradient incorrect for bottom_right pixels {p}\nis:{W.grad}\nshould be\n{bottom_right_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for bottom_right pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in mdl:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D,indxLookUpTable)
        filteredP.backward()
        assert torch.allclose(W.grad, mdl_deriv), \
                f'gradient incorrect for middle pixels {p}\nis:{W.grad}\nshould be\n{mdl_deriv}'
        assert torch.allclose(L.grad, L_mdl_deriv), \
                f'L gradient incorrect for middle pixels {p}\nis:{L.grad}\nshould be\n{L_mdl_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

# Test CoL
def test_col():
    print(f'\n\nTESTING COL')
    batches, numChannels,fmWidth = 2,5,5
    sfDepth, sfWidth    = 3,3
    k = 5
    borderSize = np.floor_divide(sfWidth,2)

    #zero input tensor
    lowerBound, upperBound = (0.0,1.0)
    bins = np.linspace(lowerBound, higherBound,k,endpoint=True)

    IT = torch.zeros((batches,numChannels,fmWidth,fmWidth), dtype=torch.float32)

    numLoops = batches*numChannels*fmWidth*fmWidth
    manNumNeighbors = sfDepth*sfWidth*sfWidth
    indxLookUpTable = genLookUpTable(IT.size(), (sfDepth,sfWidth,sfWidth))
    i = 0
    for batch in range(batches):
        for chan in range(numChannels):
            for row in range(fmWidth):
                for col in range(fmWidth):
                    #zero input tenso
                    W= torch.ones((sfDepth,sfWidth,sfWidth), requires_grad=True,dtype=torch.float32)
                    L= torch.ones((k,k), requires_grad=True,dtype=torch.float32)
                    OT = CoL(IT, W, L, bins, indxLookUpTable)
                    assert torch.allclose(OT, IT), \
                            f'CoL: IT all zeros\n{IT}, OT should be all zeros, actuall is:\n{OT}'


                    fp = OT[(batch,chan,row,col)]
                    fp.backward()
                    assert torch.allclose(W.grad, torch.zeros((sfDepth, sfWidth,sfWidth), dtype=torch.float32)), \
                            f'CoL: IT all zeros\n{IT}, all grad should be all zeros, actuall is:\n{W.grad}'
                    assert torch.allclose(L.grad, torch.zeros((k,k), dtype=torch.float32)), \
                            f'CoL: IT all zeros\n{IT}, all grad should be all zeros, actuall is:\n{L.grad}'
                    L.grad.data.zero_()
                    W.grad.data.zero_()
                    i += 1
                    #print in-place: status bar overwrite last line
                    sys.stdout.write("testing CoL progress: %d / %d  \r" % (i, numLoops))
                    sys.stdout.flush()
def test_col_viz():

    batches, numChannels,fmWidth = 1,8,8
    sfDepth, sfWidth             = 3,3
    k = 5
    borderSize = np.floor_divide(sfWidth,2)

    input('\n\ntesting COL: NO ASSERTS, JUST VISUAL CHECK')
    print(f'IT dims: ({batches}, {numChannels},{fmWidth},{fmWidth})')
    print(f'W dims: ({sfDepth}, {sfWidth}, {sfWidth})')
    print(f'L dims: ({k},{k})')

    #zero input tensor
    IT   = torch.ones((batches,numChannels,fmWidth,fmWidth), dtype=torch.float32)
    indxLookUpTable = genLookUpTable(IT.size(), (sfDepth,sfWidth,sfWidth))

    (lowerBound, UpperBound) = (0,k)
    bins = np.linspace(lowerBound, higherBound,k,endpoint=True)
    i = 0
    numLoops = batches*numChannels*fmWidth*fmWidth
    for batch in range(batches):
        for chan in range(numChannels):
            for row in range(fmWidth):
                for col in range(fmWidth):
                    #zero input tenso
                    W= torch.ones((sfDepth, sfWidth, sfWidth), requires_grad=True,dtype=torch.float32)
                    L= torch.ones((k,k), requires_grad=True,dtype=torch.float32)
                    #print(f'Spatial Filter: \n{W}')
                    #print(f'Deep CoOccur: \n{L}')
                    #print(f'Input Tensor: \n{IT}')
                    OT = CoL(IT, W, L, bins, indxLookUpTable)

                    p = (batch,chan,row,col)
                    fp = OT[(batch,chan,row,col)]
                    fp.backward()
                    #print(f'\n\nOT: \n{OT}')
                    #print(f'p: {p}')
                    #print(f'W.grad: \n{W.grad}')
                    #print(f'L.grad: \n{L.grad}')
                    L.grad.data.zero_()
                    W.grad.data.zero_()
                    i += 1
                    #print in-place: status bar overwrite last line
                    sys.stdout.write("testing CoL progress: %d / %d  \r" % (i, numLoops))
                    sys.stdout.flush()


def internal_profile_col(numTrials, batches, numChannels, fmWidth, sfDims, k):
    input('UNCOMMENT TIME RETURNS')
    sfDepth, sfWidth, _ = sfDims
    borderSize = np.floor_divide(sfWidth,2)
    W = torch.ones((sfDepth,sfWidth,sfWidth),dtype=torch.float32)
    L = torch.ones((k,k),dtype=torch.float32)
    indxLookUpTable = genLookUpTable(IT.size(), W.size())

    (lowerBound, UpperBound) = (0.0,1.0)
    bins = np.linspace(lowerBound, higherBound,k,endpoint=True)
    IT   = torch.rand((batches,numChannels,fmWidth,fmWidth), dtype=torch.float32)

    times = np.zeros((numTrials,3),dtype=np.float64)
    for i in range(numTrials):
        Timing=True
        OT, inc_times = CoL(IT, W, L, bins, indxLookUpTable)
        times[i,:] = inc_times
        #timeBin, timeUtil, timeAFRest = times[0], times[1], times[2]
        print(f'{i}th CoL run: binning {times[i,0]}, neighUtils {times[i,1]}, restAF {times[i,2]}\n')
    Timing=False
    print(times)
    times_ave, times_std = np.mean(times,axis=0), np.std(times,axis=0)
    names = ["binning","neighbor finding utils", "rest of applyFilter"]
    print(times_ave)
    for i,name in enumerate(names):
        print(f'{name}: {times_ave[i]} +- {times_std[i]}')


def profile_col(numTrials, batches, numChannels, fmWidth, sfDims, k):

    sfDepth, sfWidth, _ = sfDims
    borderSize = np.floor_divide(sfWidth,2)
    IT         = torch.rand((batches,numChannels,fmWidth,fmWidth), dtype=torch.float32)
    forward, backward  = [], []

    maxNumNeighbors = sfDepth*sfWidth*sfWidth
    indxLookUpTable = genLookUpTable((batches,numChannels,fmWidth,fmWidth), sfDims, maxNumNeighbors)
    for i in range(numTrials):
        batch = np.random.randint(0,batches)
        chan  = np.random.randint(0,numChannels)
        row   = np.random.randint(0,fmWidth)
        col   = np.random.randint(0,fmWidth)
        p = (batch,chan,row,col)

        W = torch.ones((sfDepth, sfWidth,sfWidth), requires_grad=True,dtype=torch.float32)
        L = torch.ones((k,k), requires_grad=True,dtype=torch.float32)


        with torch.autograd.profiler.profile() as prof:
            t0 = time.time()

            bnBounds = (0.0,1.0)
            OT = CoL(IT, W, L, bnBounds, indxLookUpTable)
            fp = OT[p]

            t1 = time.time()

            fp.backward()

            t2 = time.time()

            forw = t1-t0
            back = t2-t1
            forward.append(forw)
            backward.append(back)
            print(f'trial: {i}, channel: {chan}')
            print(f'forward: {forw}, backward: {back}')
            L.grad.data.zero_()
            W.grad.data.zero_()
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    input('...')
    print(f'forward: ave: {np.mean(forward)}, back: {np.mean(backward)}')
    print(f'forward: std: {np.std(forward)},  back: {np.std(backward)}')

numTrials = 5
batches, numChannels, fmWidth = 2, 128, 8
sfDepth, sfWidth = 3, 3
sfDims = (sfDepth,sfWidth,sfWidth)
k = 5

#internal_profile_col(numTrials, batches, numChannels, fmWidth, sfDims, k)
profile_col(numTrials, batches, numChannels, fmWidth, sfDims, k)
test_applyFilter()
test_col()
#test_col_viz()
