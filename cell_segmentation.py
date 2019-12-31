import numpy as np
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import distance_transform_edt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker, watershed
from scipy.signal import convolve2d
from skimage.transform import resize
from itertools import permutations
def create_tree(cellprobs):
    thresh = np.arange(0.99,0.0,-0.01)
    count=0
    leaves = []
    mergers = []
    prev = np.zeros(cellprobs.shape, dtype=int)
    fullmap = np.zeros(cellprobs.shape, dtype=int)
    D = {}
    for t in thresh:
        new, junk = label(cellprobs>t)
        if np.all(new==prev):
            continue
        idx = np.unique(new[new!=0])
        for k in idx:
            M = new==k

            oldidx = np.unique(prev[M])
            if len(oldidx)==1:
                if oldidx[0]==0:
                    #birth

                    fullmap[M] = count+1
                    count = count+1
                    new[M] = count
                    leaves.append((count,t))

            elif (len(oldidx)==2) & (0 in oldidx):
                #extension
                oldidx = oldidx[oldidx>0]
                new[M] = oldidx[0]
                fullmap[M & (prev==0)] = oldidx[0]
            else:
                #merge
                T, T_indices = distance_transform_edt(prev==0, return_indices=True)
                Tjunk = prev[T_indices[0,:,:],T_indices[1,:,:]]
                fullmap[M & (prev==0)] = Tjunk[M & (prev==0)]
                key = tuple(oldidx[oldidx>0].tolist())
                if key not in D.keys():
                    D[key] = t

        prev = fullmap.copy()
    return fullmap,  leaves


def compute_cell_bmap(cell_probs):
    output=cell_probs
    output2 = output[::2,::2]
    local_maxi = peak_local_max(output2, indices=False, min_distance=5)
    markers = ndi.label(local_maxi)[0]
    markers[output2<0.01] = -1
    segments = random_walker(output2, markers, tol=0.01)
    segments = resize(segments, output.shape, order=0, preserve_range=True)
    #segments = watershed(-output, markers)
    gx = convolve2d(segments, np.array([[1,0,-1]]), mode='same')
    gx[0,:] = 0
    gx[-1,:] = 0
    gx[:,0] = 0
    gx[:,-1] = 0
    gy = convolve2d(segments, np.array([[1,0,-1]]).T, mode='same')
    gy[0,:] = 0
    gy[-1,:] = 0
    gy[:,0] = 0
    gy[:,-1] = 0

    gmag = np.sqrt(gx**2 + gy**2)
    gmag = gmag>0
    D = {}
    P = {}
    y, x = np.where(gmag)
    for i in range(y.size):
        nearby_labels = np.unique(segments[y[i]-1:y[i]+2, x[i]-1:x[i]+2])
        t = tuple(nearby_labels)
        if t in D.keys():
            D[t].append([y[i],x[i]])
            P[t].append(np.min(cell_probs[y[i]-1:y[i]+2, x[i]-1:x[i]+2]))
        else:
            D[t] = [[y[i],x[i]]]
            P[t] = [np.min(cell_probs[y[i]-1:y[i]+2, x[i]-1:x[i]+2])]
    bmap = np.zeros(cell_probs.shape)
    for t in D.keys():
        coords = np.array(D[t])

        #if 2-way boundary:
        if len(t)<3:
            score = np.mean(np.array(P[t]))
        else:
            perms = permutations(t, 2)
            perms = [np.mean(P[t]) for t in perms if t in P.keys()]
            score = np.min(perms)
        bmap[coords[:,0],coords[:,1]] = 1-score
    bmap[0,:] = 1
    bmap[-1,:] = 1
    bmap[:,0] = 1
    bmap[:,-1] = 1
    return bmap

def get_cells(cell_probs, bmap, boundary_thresh, cell_thresh=0.001):
    seg = bmap < boundary_thresh
    labels, junk = label(seg)
    totalprobs = np.zeros(np.max(labels)+1)
    count = np.zeros(np.max(labels)+1)
    np.add.at(totalprobs, labels.reshape(-1), cell_probs.reshape(-1))
    np.add.at(count, labels.reshape(-1), 1)
    avgprobs = totalprobs/count
    labels_to_zero = np.where(avgprobs<cell_thresh)[0]
    for l in labels_to_zero:
        labels[labels==l] = 0
    return labels
