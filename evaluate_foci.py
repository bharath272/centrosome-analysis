import numpy as np
import json
import os
from sklearn.metrics import pairwise_distances

def evaluate_single(annotfile, detfile, dist_thresh=2, allow_duplicates=False):
    np.set_printoptions(suppress=True)
    with open(annotfile,'r') as f:
        annots = json.load(f)
    gtfoci = annots['foci']
    for a in gtfoci:
        if [] in a:
            a.remove([])

    gtfoci = [np.array(x) for x in gtfoci if len(x)>0]
    if len(gtfoci)==0:
        return [],[],[],0

    gtfoci = np.concatenate(gtfoci, axis=0)
    print(gtfoci)

    with open(detfile, 'r') as f:
        dets = json.load(f)
    foci = dets['foci']

    foci_scores = dets['foci_scores']
    dist = pairwise_distances(gtfoci, foci)

    mindist = np.min(dist, axis=0)

    argmindist = np.argmin(dist,axis=0)
    labels = mindist<dist_thresh
    argmindist[~labels] = -1
    if not allow_duplicates:
        for i in range(argmindist.size):
            if i>0:
                if argmindist[i] in argmindist[:i].tolist():
                    labels[i] = False
                    argmindist[i] = -1
    return foci_scores, labels, argmindist, gtfoci.shape[0]

def compute_ap(prec,rec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre

def evaluate_PR(scores, labels, idsdet,total):

    scores = np.array(scores)
    labels = np.array(labels)
    idx = np.argsort(-scores)
    scores = scores[idx]
    labels = labels[idx]
    idsdet = [idsdet[i] for i in idx]
    print(labels[:3])
    print(idsdet[:3])
    tp = np.cumsum(labels).astype(float)
    fp = np.cumsum(~labels).astype(float)
    prec = tp/(tp+fp)
    rec = np.zeros_like(prec)
    for i in range(prec.size):
        idsdet2 = [x for j,x in enumerate(idsdet[:(i+1)]) if x[1]!=-1]
        idsdet2 = list(set(idsdet2))
        rec[i] = len(idsdet2)/float(total)
    return prec,rec

def evaluate_all(rootdir, detdir, files, dist_thresh=2, allow_duplicates=False):
    annotfiles = [os.path.join(rootdir, 'annots',x+'.json') for x in files]
    detfiles = [os.path.join(detdir, x+'.json') for x in files]
    foci_scores = []
    labels = []
    gtids = []
    totalgtfoci = 0
    for i in range(len(annotfiles)):
        s,l,ids,gtc = evaluate_single(annotfiles[i],detfiles[i],dist_thresh=dist_thresh, allow_duplicates=allow_duplicates)
        ids = [(i,x) for x in ids]
        foci_scores.extend(s)
        labels.extend(l)
        gtids.extend(ids)
        totalgtfoci = totalgtfoci + gtc

    prec, rec = evaluate_PR(foci_scores, labels, gtids, totalgtfoci)
    ap, mprec = compute_ap(prec, rec)
    return prec, rec, ap, mprec[1:-1]
