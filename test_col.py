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

    IT = torch.zeros((1,3,5,5), dtype=torch.float64)
    IT[:,0,:,:], IT[:,1,:,:], IT[:,2,:,:]  = 1.0, 2.0, 3.0
    W = torch.ones((3,3,3), requires_grad=True,dtype=torch.float64)
    L = torch.ones((k,k), requires_grad=True,dtype=torch.float64)
    bins = np.arange(.5,4.5,1)
    IT_Binned = np.digitize(IT,bins) - 1

    #print(f'Spatial Filter: \n{W}')
    #print(f'Input Tensor: \n{IT}')
    #print(f'Deep CoOccur: \n{L}')
    #print(f'Input Ten Bin:\n{IT_Binned}')
    ls,rs,ts,bs,mdl            = genBorderIndeces(fmWidth, borderSize)
    lt, lb, rt, rb, l, r, t, b = makeDisjoint(ls,rs,ts,bs,mdl)

    #MIDDLE CHANNEL
    #Only do 1st (middle) Channel atm

    #TODO: delete requires_grad from *_deriv

    #define derivs
    left_deriv = torch.ones((3,3,3), dtype=torch.float64)
    left_deriv[1,:,:], left_deriv[2,:,:] = 2, 3
    left_deriv[:,:,0]=0.0

    right_deriv = torch.ones((3,3,3), dtype=torch.float64)
    right_deriv[1,:,:], right_deriv[2,:,:] = 2,3
    right_deriv[:,:,2]=0.0

    top_deriv = torch.ones((3,3,3), dtype=torch.float64)
    top_deriv[1,:,:], top_deriv[2,:,:] = 2, 3
    top_deriv[:,0,:]=0.0

    bottom_deriv = torch.ones((3,3,3), dtype=torch.float64)
    bottom_deriv[1,:,:], bottom_deriv[2,:,:] = 2, 3
    bottom_deriv[:,2,:]=0.0

    L_side_deriv = torch.zeros((k,k), requires_grad=True,dtype=torch.float64)
    L_side_deriv[1,0], L_side_deriv[1,1],  L_side_deriv[1,2] = 6*1,6*2,6*3


    top_left_deriv = torch.ones((3,3,3), dtype=torch.float64)
    top_left_deriv[1,:,:], top_left_deriv[2,:,:] = 2, 3
    top_left_deriv[:,:,0], top_left_deriv[:,0,:] = 0.0, 0.0

    top_right_deriv = torch.ones((3,3,3), dtype=torch.float64)
    top_right_deriv[1,:,:], top_right_deriv[2,:,:] = 2, 3
    top_right_deriv[:,:,2], top_right_deriv[:,0,:] = 0.0, 0.0

    bottom_left_deriv = torch.ones((3,3,3), dtype=torch.float64)
    bottom_left_deriv[1,:,:], bottom_left_deriv[2,:,:] = 2, 3
    bottom_left_deriv[:,:,0], bottom_left_deriv[:,2,:] = 0.0, 0.0

    bottom_right_deriv = torch.ones((3,3,3), dtype=torch.float64)
    bottom_right_deriv[1,:,:], bottom_right_deriv[2,:,:] = 2, 3
    bottom_right_deriv[:,2,:], bottom_right_deriv[:,:,2] = 0.0, 0.0

    L_corner_deriv = torch.zeros((k,k), requires_grad=True,dtype=torch.float64)
    L_corner_deriv[1,0], L_corner_deriv[1,1],  L_corner_deriv[1,2] = 4*1,4*2,4*3


    mdl_deriv = torch.ones((3,3,3), dtype=torch.float64)
    mdl_deriv[1,:,:], mdl_deriv[2,:,:] = 2, 3

    L_mdl_deriv = torch.zeros((k,k), requires_grad=True,dtype=torch.float64)
    L_mdl_deriv[1,0], L_mdl_deriv[1,1],  L_mdl_deriv[1,2] = 9*1,9*2,9*3

    chan = 2-1

    print(f'Testing W gradients:fmWidth: {fmWidth}, sfWidth: {sfWidth}')
    for p in l:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, left_deriv), \
                f'gradient incorrect for left pixels {p}\nis:{W.grad}\nshould be\n{left_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for left pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in r:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, right_deriv), \
                f'gradient incorrect for right pixels {p}\nis:{W.grad}\nshould be\n{right_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for right pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in t:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, top_deriv), \
                f'gradient incorrect for top pixels {p}\nis:{W.grad}\nshould be\n{top_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for right pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in b:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_deriv), \
                f'gradient incorrect for bottom pixels {p}\nis:{W.grad}\nshould be\n{bottom_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for bottom pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in lt:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, top_left_deriv), \
                f'gradient incorrect for top_left pixels {p}\nis:{W.grad}\nshould be\n{top_left_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for top_left pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in lb:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_left_deriv), \
                f'gradient incorrect for bottom_left pixels {p}\nis:{W.grad}\nshould be\n{bottom_left_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for bottom_left pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in rt:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, top_right_deriv), \
                f'gradient incorrect for top_right pixels {p}\nis:{W.grad}\nshould be\n{top_right_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for top_right pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()


    for p in rb:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_right_deriv), \
                f'gradient incorrect for bottom_right pixels {p}\nis:{W.grad}\nshould be\n{bottom_right_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for bottom_right pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in mdl:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
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
    left_deriv = torch.ones((3,3,3), dtype=torch.float64)
    left_deriv[2,:,:], left_deriv[0,:,:] = 2, 3
    left_deriv[:,:,0]=0.0

    right_deriv = torch.ones((3,3,3), dtype=torch.float64)
    right_deriv[2,:,:], right_deriv[0,:,:] = 2,3
    right_deriv[:,:,2]=0.0

    top_deriv = torch.ones((3,3,3), dtype=torch.float64)
    top_deriv[2,:,:], top_deriv[0,:,:] = 2, 3
    top_deriv[:,0,:]=0.0

    bottom_deriv = torch.ones((3,3,3), dtype=torch.float64)
    bottom_deriv[2,:,:], bottom_deriv[0,:,:] = 2, 3
    bottom_deriv[:,2,:]=0.0

    L_side_deriv = torch.zeros((k,k), requires_grad=True,dtype=torch.float64)
    L_side_deriv[0,0], L_side_deriv[0,1],  L_side_deriv[0,2] = 6*1,6*2,6*3


    top_left_deriv = torch.ones((3,3,3), dtype=torch.float64)
    top_left_deriv[2,:,:], top_left_deriv[0,:,:] = 2, 3
    top_left_deriv[:,:,0], top_left_deriv[:,0,:] = 0.0, 0.0

    top_right_deriv = torch.ones((3,3,3), dtype=torch.float64)
    top_right_deriv[2,:,:], top_right_deriv[0,:,:] = 2, 3
    top_right_deriv[:,:,2], top_right_deriv[:,0,:] = 0.0, 0.0

    bottom_left_deriv = torch.ones((3,3,3), dtype=torch.float64)
    bottom_left_deriv[2,:,:], bottom_left_deriv[0,:,:] = 2, 3
    bottom_left_deriv[:,:,0], bottom_left_deriv[:,2,:] = 0.0, 0.0

    bottom_right_deriv = torch.ones((3,3,3), dtype=torch.float64)
    bottom_right_deriv[2,:,:], bottom_right_deriv[0,:,:] = 2, 3
    bottom_right_deriv[:,2,:], bottom_right_deriv[:,:,2] = 0.0, 0.0

    L_corner_deriv = torch.zeros((k,k), requires_grad=True,dtype=torch.float64)
    L_corner_deriv[0,0], L_corner_deriv[0,1],  L_corner_deriv[0,2] = 4*1,4*2,4*3


    mdl_deriv = torch.ones((3,3,3), dtype=torch.float64)
    mdl_deriv[2,:,:], mdl_deriv[0,:,:] = 2, 3

    L_mdl_deriv = torch.zeros((k,k), requires_grad=True,dtype=torch.float64)
    L_mdl_deriv[0,0], L_mdl_deriv[0,1],  L_mdl_deriv[0,2] = 9*1,9*2,9*3

    for p in l:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, left_deriv), \
                f'gradient incorrect for left pixels {p}\nis:{W.grad}\nshould be\n{left_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for left pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in r:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, right_deriv), \
                f'gradient incorrect for right pixels {p}\nis:{W.grad}\nshould be\n{right_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for right pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in t:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, top_deriv), \
                f'gradient incorrect for top pixels {p}\nis:{W.grad}\nshould be\n{top_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for right pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in b:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_deriv), \
                f'gradient incorrect for bottom pixels {p}\nis:{W.grad}\nshould be\n{bottom_deriv}'
        assert torch.allclose(L.grad, L_side_deriv), \
                f'L gradient incorrect for bottom pixels {p}\nis:{L.grad}\nshould be\n{L_side_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in lt:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, top_left_deriv), \
                f'gradient incorrect for top_left pixels {p}\nis:{W.grad}\nshould be\n{top_left_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for top_left pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in lb:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_left_deriv), \
                f'gradient incorrect for bottom_left pixels {p}\nis:{W.grad}\nshould be\n{bottom_left_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for bottom_left pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in rt:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, top_right_deriv), \
                f'gradient incorrect for top_right pixels {p}\nis:{W.grad}\nshould be\n{top_right_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for top_right pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()


    for p in rb:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
        filteredP.backward()
        assert torch.allclose(W.grad, bottom_right_deriv), \
                f'gradient incorrect for bottom_right pixels {p}\nis:{W.grad}\nshould be\n{bottom_right_deriv}'
        assert torch.allclose(L.grad, L_corner_deriv), \
                f'L gradient incorrect for bottom_right pixels {p}\nis:{L.grad}\nshould be\n{L_corner_deriv}'
        L.grad.data.zero_()
        W.grad.data.zero_()

    for p in mdl:
        p_4D = (0,chan,p[0],p[1])
        filteredP  = applyFilter(IT,IT_Binned,W,L,p_4D)
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
    batches, numChannels,fmWidth = 1,3,5
    sfDepth, sfWidth    = 3,3
    k = 5
    borderSize = np.floor_divide(sfWidth,2)

    #zero input tensor
    """
    IT = torch.zeros((1,numChannels,fmWidth,fmWidth), dtype=torch.float64)
    W  = torch.ones((sfWidth,sfWidth,sfDepth), requires_grad=True,dtype=torch.float64)
    L  = torch.ones((k,k), requires_grad=True,dtype=torch.float64)
    print(f'Spatial Filter: \n{W}')
    print(f'Deep CoOccur: \n{L}')
    print(f'Input Tensor: \n{IT}')
    OT = CoL(IT, W, L)
    ls,rs,ts,bs,mdl            = genBorderIndeces(fmWidth, borderSize)
    lt, lb, rt, rb, l, r, t, b = makeDisjoint(ls,rs,ts,bs,mdl)

    assert torch.allclose(OT, torch.zeros((1,numChannels,fmWidth,fmWidth), dtype=torch.float64)), \
            f'CoL: IT all zeros\n{IT}, OT should be all zeros, actuall is:\n{OT}'
    """

    IT = torch.zeros((1,numChannels,fmWidth,fmWidth), dtype=torch.float64)

    numLoops = batches*numChannels*fmWidth*fmWidth
    i = 0
    for batch in range(batches):
        for chan in range(numChannels):
            for row in range(fmWidth):
                for col in range(fmWidth):
                    #zero input tenso
                    W= torch.ones((sfWidth,sfWidth,sfDepth), requires_grad=True,dtype=torch.float64)
                    L= torch.ones((k,k), requires_grad=True,dtype=torch.float64)
                    #print(f'Spatial Filter: \n{W}')
                    #print(f'Deep CoOccur: \n{L}')
                    #print(f'Input Tensor: \n{IT}')
                    OT = CoL(IT, W, L)

                    fp = OT[(batch,chan,row,col)]
                    fp.backward()
                    assert torch.allclose(W.grad, torch.zeros((sfWidth,sfWidth,sfDepth), dtype=torch.float64)), \
                            f'CoL: IT all zeros\n{IT}, all grad should be all zeros, actuall is:\n{W.grad}'
                    assert torch.allclose(L.grad, torch.zeros((k,k), dtype=torch.float64)), \
                            f'CoL: IT all zeros\n{IT}, all grad should be all zeros, actuall is:\n{L.grad}'
                    L.grad.data.zero_()
                    W.grad.data.zero_()
                    i += 1
                    #print in-place: status bar overwrite last line
                    sys.stdout.write("testing CoL progress: %d / %d  \r" % (i, numLoops))
                    sys.stdout.flush()


test_applyFilter()
test_col()
