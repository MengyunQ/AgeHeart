import matplotlib.pyplot as plt
import numpy as np

from scipy import interpolate

def sa2la_slice(sa, sa_affine, la, la_affine, method='nearest'):
    # transfrom SA [i, j, k] to LA [I, J, K] space
    # [I, J, K]' = inv(R_la)*R_sa*[i, j, k]' + inv(R_la)*(b_sa-b_la) = A*[i, j, k]' + b
    R_la, b_la = la_affine[0:3, 0:3], la_affine[0:3, -1]
    R_sa, b_sa = sa_affine[0:3, 0:3], sa_affine[0:3, -1]
    A = np.linalg.inv(R_la).dot(R_sa)
    b = np.linalg.inv(R_la).dot(b_sa-b_la)

    sa_size = sa.shape
    Grid_i, Grid_j, Grid_k = np.meshgrid(range(0, sa_size[0]), range(0, sa_size[1]), range(0, sa_size[2]), indexing='ij')
    LA_coor = A.dot(np.array([Grid_i.reshape(-1), Grid_j.reshape(-1), Grid_k.reshape(-1)])) + b.reshape((3,1))
    Grid_I, Grid_J, Grid_K = LA_coor[0].reshape(sa_size), LA_coor[1].reshape(sa_size), LA_coor[2].reshape(sa_size)

    SA_grid = np.array([Grid_I.reshape(-1), Grid_J.reshape(-1), Grid_K.reshape(-1)]).T
    SA_label = sa.reshape(-1)
    la_size = la.shape
    LA_I, LA_J, LA_K = np.meshgrid(range(0, la_size[0]), range(0, la_size[1]), range(0, la_size[2]), indexing='ij')
    LA_grid = np.array([LA_I.reshape(-1), LA_J.reshape(-1), LA_K.reshape(-1)]).T
    LA_label = interpolate.griddata(SA_grid, SA_label, LA_grid, method=method)

    return LA_label.reshape(la_size)


def mosaic(vol, size=[10, 10], cmap='viridis', clim=[0, 4]):
    "input vol like: z x 128 x 128 "

    plt.style.use('dark_background')
    fig, axs = plt.subplots(nrows=size[0], ncols=size[1], frameon=False,
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    fig.set_size_inches([size[1], size[0]])
    [ax.set_axis_off() for ax in axs.ravel()]

    for k in range(0, min(size[0]*size[1], vol.shape[0])):
        m = k // size[1]
        n = k - m*size[1]
        axs[m, n].imshow(vol[k, :, :], cmap=cmap, clim=clim)

    return fig



''' image preprocessing to aligh with HR volume'''
def align_GENvolume(input_vol, affine_matrix = None):
    "input GENScan volume: 256 x 256 x z, output z x 256 x 256"
    output = np.flip(input_vol.squeeze().transpose((2, 1, 0)), axis=1)

    if affine_matrix is None:
        return output
    else:
        R, b = affine_matrix[0:3, 0:3], affine_matrix[0:3, -1]
        T = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        bT = np.array([0, output.shape[1]-1, 0])

        R_new = R.dot(T)
        b_new = - R.dot(T).dot(bT) + b
        new_matrix = affine_matrix.copy()
        new_matrix[0:3, 0:3] = R_new
        new_matrix[0:3, -1] = b_new
        return output, new_matrix

def align_UKBvolume(input_vol, affine_matrix=None):
    "input UKBScan volume like: 204 x 208 x 13, output 13 x 208 x 204"
    output = np.flip(input_vol.squeeze().transpose((2, 1, 0)), axis=0)
    ## rotate affine
    ## center
    if affine_matrix is None:
        return output
    else:
        R, b = affine_matrix[0:3, 0:3], affine_matrix[0:3, -1]
        T = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        bT = np.array([output.shape[0]-1, 0, 0])

        R_new = R.dot(T)
        b_new = - R.dot(T).dot(bT) + b
        new_matrix = affine_matrix.copy()
        new_matrix[0:3, 0:3] = R_new
        new_matrix[0:3, -1] = b_new
        return output, new_matrix

def crop_3Dimage(image, center, size, affine_matrix=None):
    """ Crop a 3D image using a bounding box centred at (c0, c1, c2) with specified size (size0, size1, size2) """
    c0, c1, c2 = center
    size0, size1, size2 = size

    S0, S1, S2 = image.shape

    r0, r1, r2 = int(size0 / 2), int(size1 / 2), int(size2 / 2)
    start0, end0 = c0 - r0, c0 + r0
    start1, end1 = c1 - r1, c1 + r1
    start2, end2 = c2 - r2, c2 + r2

    start0_, end0_ = max(start0, 0), min(end0, S0)
    start1_, end1_ = max(start1, 0), min(end1, S1)
    start2_, end2_ = max(start2, 0), min(end2, S2)

    # Crop the image
    crop = image[start0_: end0_, start1_: end1_, start2_: end2_]
    crop = np.pad(crop,
                  ((start0_ - start0, end0 - end0_), (start1_ - start1, end1 - end1_), (start2_ - start2, end2 - end2_)),
                  'constant')

    if affine_matrix is None:
        return crop
    else:
        R, b = affine_matrix[0:3, 0:3], affine_matrix[0:3, -1]
        affine_matrix[0:3, -1] = R.dot(np.array([c0-r0, c1-r1, c2-r2])) + b
        return crop, affine_matrix


''' transform between labelmap and one-hot encoding'''
def onehot2label(seg_onehot, axis=0):
    "input Cx64x128x128 volume, output 64x128x128, 0 - bg, 1-LV, 2-MYO, 4-RV"
    labelmap = np.argmax(seg_onehot, axis=axis)
    tmplabel = labelmap.copy()
    labelmap[tmplabel == 0] = 0
    labelmap[tmplabel == 1] = 1
    labelmap[tmplabel == 2] = 2
    labelmap[tmplabel == 3] = 4
    return labelmap


def label2onehot(labelmap):
    "input 64x128x128 volume, output Cx64x128x128"
    seg_onehot = []
    seg_onehot.append([labelmap == 0])
    seg_onehot.append([labelmap == 1])
    seg_onehot.append([labelmap == 2])
    seg_onehot.append([labelmap == 4])
    return np.concatenate(seg_onehot, axis=0)


''' calculate dice'''
def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))

def np_mean_dice(pred, truth):
    """ Dice mean metric """
    dsc = []
    for k in np.unique(truth)[1:]:
        dsc.append(np_categorical_dice(pred, truth, k))
    return np.array(dsc)


''' slice volume and plot'''
def vol3view(vol, clim=(0,4), cmap='viridis'):
    " input volume: 64 x 128 x 128"
    plt.style.use('default')
    #plt.style.use('dark_background')
    fig, axs = plt.subplots(nrows=3, ncols=1, frameon=False,
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    fig.set_size_inches([1, 3])
    [ax.set_axis_off() for ax in axs.ravel()]

    view1 = vol[32, :, :]
    view2 = vol[:, 64, :]
    view3 = vol[:, :, 64]

    axs[0].imshow(view1, clim=clim, cmap=cmap)
    axs[1].imshow(view2, clim=clim, cmap=cmap)
    axs[2].imshow(view3, clim=clim, cmap=cmap)

    return fig



def move_3Dimage(image, d):
        """  """
        d0, d1, d2 = d
        S0, S1, S2 = image.shape

        start0, end0 = 0 - d0, S0 - d0
        start1, end1 = 0 - d1, S1 - d1
        start2, end2 = 0 - d2, S2 - d2

        start0_, end0_ = max(start0, 0), min(end0, S0)
        start1_, end1_ = max(start1, 0), min(end1, S1)
        start2_, end2_ = max(start2, 0), min(end2, S2)

        # Crop the image
        crop = image[start0_: end0_, start1_: end1_, start2_: end2_]
        crop = np.pad(crop,
                      ((start0_ - start0, end0 - end0_), (start1_ - start1, end1 - end1_),
                       (start2_ - start2, end2 - end2_)),
                      'constant')

        return crop


def np_categorical_dice_optim(seg,gt,k=1,ds=None):
    if ds is not None:
        _, dh, dw = ds
        seg_1 = move_3Dimage(seg, (0, dh, dw))
        dice_1 = np_categorical_dice(seg_1, gt, k)
        return dice_1

    else:
        d = 5
        max_dice = 0
        best_dh, best_dw = 0, 0
        for dh in range(-d, d):
            for dw in range(-d, d):
                seg_1 = move_3Dimage(seg, (0, dh, dw))
                dice_1 = np_categorical_dice(seg_1, gt, k)
                if dice_1 > max_dice:
                    best_dh, best_dw = dh, dw
                    max_dice = dice_1
        ds = (0, best_dh, best_dw)
        return max_dice, ds


def np_mean_dice_optim(pred, truth, ds):
    """ Dice mean metric """
    dsc = []
    for k in np.unique(truth)[1:]:
        dsc.append(np_categorical_dice_optim(pred, truth, k, ds=ds))
    return np.mean(dsc)


from skimage.measure import label
def clean_seg(seg):
    labels = label(seg>0)
    assert(labels.max() != 0) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return seg*largestCC

def find_ES(seg):
    ## calculate the lowest volume
    #dice_seq = []
    LVV = []
    for s in range(0, seg. shape[0]):
        seg_s = seg[s]
        LVV.append(np.sum(seg_s == 1))
        #dsc = np_mean_dice(seg_s, seg_ed)
    index = LVV.index(min(LVV))
    return seg[index, ...], index

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list