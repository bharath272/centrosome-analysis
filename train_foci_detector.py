import ml_foci_detect
import argparse
import os
import torch
def parse_args():
    parser = argparse.ArgumentParser(description='Train Foci Detector')
    parser.add_argument('--trainfiles', help='csv containing list of training images and corresponding annotations', required=True)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--learningrate', type=float, default=0.01)
    parser.add_argument('--weightdecay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--savecheckpoints', action='store_true')
    parser.add_argument('--checkpointdir', type=str, default='checkpoints')
    parser.add_argument('--modelfile', type=str, default='foci_model.pt')
    return parser.parse_args()

if __name__=='__main__':
    params = parse_args();
    with open(params.trainfiles, 'r') as f:
        trainfiles = f.readlines()
    trainfiles = [x.strip().split(',') for x in trainfiles]

    if not os.path.isdir(params.checkpointdir):
        os.makedirs(params.checkpointdir)

    model = ml_foci_detect.MultiChannelCombinedScorer()
    model = ml_foci_detect.train_model_fcn(model, trainfiles, batchsize=params.batchsize, need_sigmoid=False, \
     lr=params.learningrate, momentum=params.momentum, weight_decay=params.weightdecay, \
     num_epochs=params.epochs,checkpointdir=params.checkpointdir, savecheckpoints=params.savecheckpoints)
    torch.save(model.state_dict(), params.modelfile)
