import ml_foci_detect
import evaluate_foci
import argparse
import os
import torch
import json
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Test Foci Detector')
    parser.add_argument('--testfiles', help='csv containing list of testing images and corresponding annotations', required=True)
    parser.add_argument('--outputdir', type=str, default='outputs')
    parser.add_argument('--modelfile', type=str, default='foci_model.pt')
    return parser.parse_args()

if __name__=='__main__':
    params = parse_args();
    with open(params.testfiles, 'r') as f:
        testfiles = f.readlines()
    testfiles = [x.strip().split(',') for x in testfiles[1:]]
    #testfiles = [testfiles[i] for i in [0,2,3,4]]
    testimgs = [x[0] for x in testfiles]
    testannots = [x[1] for x in testfiles]

    if not os.path.isdir(params.outputdir):
        os.makedirs(params.outputdir)

    model = ml_foci_detect.MultiChannelCombinedScorer()
    modelparams = torch.load(params.modelfile)
    model.load_state_dict(modelparams)
    ml_foci_detect.save_dets(model, testimgs, params.outputdir)


    stems = [os.path.splitext(os.path.basename(x))[0] for x in testimgs]
    detfiles = [os.path.join(params.outputdir, x+'.json') for x in stems]
    prec, rec, ap, mprec = evaluate_foci.evaluate_all(testannots, detfiles, dist_thresh=10)
    np.savez(os.path.join(params.outputdir, 'results.npz'), prec=mprec,rec=rec,ap=ap)
    print(ap)
