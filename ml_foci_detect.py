import torch
import numpy as np
import torch.nn as nn
import os
import json
import tifffile
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
from scipy.ndimage.measurements import label
from scipy.signal import medfilt
from sklearn.metrics import pairwise_distances
import evaluate_foci
from skimage.transform import resize
import scale_estimator

def normalize_for_display(img):
    img = img.astype(float).transpose(1,2,0)
    for i in range(img.shape[2]):
        img[:,:,i] = (img[:,:,i] - np.min(img[:,:,i]))/(np.max(img[:,:,i])-np.min(img[:,:,i])+1e-5)
    return img

def crop_patch_disp(img, patch):
    """
    Args:
        [img]   Shape:HxW   Grayscale image.
        [xmin]  Int         Minimum index on x-axis (i.e., along the width)
        [xmax]  Inta        Maximum index on x-axis (i.e., alone the width)
        [ymin]  Int         Minimum index on y-axis (i.e., along the height)
        [ymax]  Int         Minimum index on y-axis (i.e., along the height)
    Rets:
        Image of shape up to (ymax-ymin, xmax-xmin).
        If the index range goes outside the image, the return patch will be cropped.
    """
    xmin = int(patch[0])
    ymin = int(patch[1])
    xmax = int(patch[2])
    ymax = int(patch[3])
    newimg = np.zeros((ymax-ymin, xmax-xmin, img.shape[2]))
    lby = np.maximum(ymin, 0)
    uby = np.minimum(ymax, img.shape[0])
    lbx = np.maximum(xmin, 0)
    ubx = np.minimum(xmax, img.shape[1])
    newimg[lby-ymin:uby-ymin, lbx-xmin:ubx-xmin,:] = img[lby:uby, lbx:ubx,:]
    return newimg
def merge_annots(annots):
    gtfoci = annots['foci']
    for a in gtfoci:
        if [] in a:
            a.remove([])

    gtfoci = [np.array(x) for x in gtfoci if len(x)>0]
    return np.concatenate(gtfoci,axis=0)
class FociDetector(nn.Module):
    def __init__(self, input_channels=3, input_size=17, ksize=5,hidden_channels=10):
        super(FociDetector,self).__init__()
        self.conv1 = nn.Conv2d(input_channels,hidden_channels,ksize,stride=2, padding=int((ksize-1)/2))
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels,ksize, stride=2, padding=int((ksize-1)/2))
        self.conv3 = nn.Conv2d(hidden_channels,hidden_channels,ksize, stride=2, padding=int((ksize-1)/2))
        self.finalmapsize = ksize #input_size-(ksize-1)*3


        self.convf = nn.Conv2d(hidden_channels, 1, self.finalmapsize, padding=int((ksize-1)/2))

        self.relu = nn.ReLU()

    def forward(self,x):
        output = self.relu(self.conv1(x))

        output = self.relu(self.conv2(output))

        output = self.relu(self.conv3(output))

        output = self.convf(output)

        return output





class MultiChannelFociDetector(nn.Module):
    def __init__(self, input_size=17, ksize=5, hidden_channels=10):
        super(MultiChannelFociDetector,self).__init__()
        self.channel1 = FociDetector(input_channels=1,input_size=input_size, ksize=ksize, hidden_channels=hidden_channels)
        self.channel2 = FociDetector(input_channels=1,input_size=input_size, ksize=ksize, hidden_channels=hidden_channels)
        self.scale_estimator = scale_estimator.ScaleEstimator()
    def forward(self, x):
        scale, bias = self.scale_estimator(x)
        x = x*scale+bias
        output1 = self.channel1(x[:,[0],:,:])
        output2 = self.channel2(x[:,[1],:,:])
        return torch.min(output1, output2)

class MultiChannelSimpleScorer(nn.Module):
    def __init__(self):
        super(MultiChannelSimpleScorer, self).__init__()
        self.scale_estimator = scale_estimator.ScaleEstimator()

    def forward(self, x):
        scale, bias = self.scale_estimator(x)
        x = x*scale + bias
        return torch.min(x[:,[0],:,:], x[:,[1],:,:])
class MultiChannelCombinedScorer(nn.Module):
    def __init__(self,input_size=17, ksize=5, hidden_channels=10):
        super(MultiChannelCombinedScorer, self).__init__()
        #self.scale_estimator = scale_estimator.ScaleEstimator()
        self.channel1 = FociDetector(input_channels=1,input_size=input_size, ksize=ksize, hidden_channels=hidden_channels)
        self.channel2 = FociDetector(input_channels=1,input_size=input_size, ksize=ksize, hidden_channels=hidden_channels)
        #self.upsample = nn.Upsample(scale_factor=8, mode='nearest')
    def forward(self,x):

        #upsample = nn.Upsample(size=(x.shape[2],x.shape[3]))
        #scale, bias = self.scale_estimator(x)
        #print("scale=", scale)
        #print("bias=", bias)
        #x = x*scale+bias
        output1 = torch.sigmoid(F.interpolate(self.channel1(x[:,[0],:,:]), size=(x.shape[2],x.shape[3])))
        output2 = torch.sigmoid(F.interpolate(self.channel2(x[:,[1],:,:]), size=(x.shape[2],x.shape[3])))
        output3 = torch.sigmoid(x[:,[0],:,:])
        output4 = torch.sigmoid(x[:,[1],:,:])
        return output1*output2*output3*output4
    def forward_vis(self, x):
        output1 = torch.sigmoid(F.interpolate(self.channel1(x[:,[0],:,:]), size=(x.shape[2],x.shape[3])))
        output2 = torch.sigmoid(F.interpolate(self.channel2(x[:,[1],:,:]), size=(x.shape[2],x.shape[3])))
        output3 = torch.sigmoid(x[:,[0],:,:])
        output4 = torch.sigmoid(x[:,[1],:,:])
        return output1, output2, output3, output4


def get_positive_patches(point_list, half_patch_size):

    mins = np.round(point_list-half_patch_size).astype(int)
    maxes = mins+2*half_patch_size+1
    patches = np.concatenate((mins, maxes), axis=1)
    return patches


def get_negative_patches(image_region, num, half_patch_size):
    image_region = np.array(image_region).reshape((1,4))
    center = np.random.rand(num,2)
    center = image_region[:,:2] + center*(image_region[:,2:]-image_region[:,:2])
    minpatch = np.round(center - half_patch_size).astype(int)
    maxpatch = minpatch + 2*half_patch_size+1
    patches = np.concatenate((minpatch, maxpatch),axis=1)
    return patches

def crop_patch(img, patch):
    """
    Args:
        [img]   Shape:HxW   Grayscale image.
        [xmin]  Int         Minimum index on x-axis (i.e., along the width)
        [xmax]  Inta        Maximum index on x-axis (i.e., alone the width)
        [ymin]  Int         Minimum index on y-axis (i.e., along the height)
        [ymax]  Int         Minimum index on y-axis (i.e., along the height)
    Rets:
        Image of shape up to (ymax-ymin, xmax-xmin).
        If the index range goes outside the image, the return patch will be cropped.
    """
    xmin = int(patch[0])
    ymin = int(patch[1])
    xmax = int(patch[2])
    ymax = int(patch[3])
    newimg = np.zeros((img.shape[0],ymax-ymin, xmax-xmin))
    lby = np.maximum(ymin, 0)
    uby = np.minimum(ymax, img.shape[1])
    lbx = np.maximum(xmin, 0)
    ubx = np.minimum(xmax, img.shape[2])
    newimg[:,lby-ymin:uby-ymin, lbx-xmin:ubx-xmin] = img[:,lby:uby, lbx:ubx]
    return newimg


class FociPatchDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, train_files,half_patch_size=8, pos_frac = 0.25, mean_patch=None, std_patch=None):
        super(FociPatchDataset, self).__init__()
        train_imgs = [os.path.join(rootdir, 'imgs', x+'.tif') for x in train_files]
        train_annots = [os.path.join(rootdir, 'annots', x+'.json') for x in train_files]
        patches = []
        labels = []
        image_ids = []
        for i, e in enumerate(train_annots):
            with open(e,'r') as f:
                annot = json.load(f)

            foci = annot['foci']
            pos_patches_this = []
            neg_patches_this = []
            for x in foci:
                if [] in x:
                    x.remove([])
                if len(x)==0:
                    continue
                x = np.array(x)
                patches_tmp = get_positive_patches(x, half_patch_size)
                pos_patches_this.append(patches_tmp)
            if len(pos_patches_this)==0:

                continue

            pos_patches_this = np.concatenate(pos_patches_this, axis=0)
            neg_patches_this.append(pos_patches_this + np.array([[-4,0,-4,0]]))
            neg_patches_this.append(pos_patches_this + np.array([[4,0,4,0]]))
            neg_patches_this.append(pos_patches_this + np.array([[0,-4,0,-4]]))
            neg_patches_this.append(pos_patches_this + np.array([[0,4,0,4]]))

            img = tifffile.imread(train_imgs[i])
            imshape = img[0,:,:].shape
            neg_patches_this.append(get_negative_patches([0,0,imshape[1],imshape[0]],pos_patches_this.shape[0]*4, half_patch_size))
            neg_patches_this = np.concatenate(neg_patches_this, axis=0)
            num_neg = int(pos_patches_this.shape[0]*(1-pos_frac)/pos_frac)
            neg_patches_this = neg_patches_this[np.random.choice(neg_patches_this.shape[0],num_neg),:]


            patches_this = np.concatenate((pos_patches_this, neg_patches_this), axis=0)
            labels_this = np.concatenate((np.ones(pos_patches_this.shape[0]), np.zeros(neg_patches_this.shape[0])))
            image_ids_this = np.ones(patches_this.shape[0], dtype='int32')*i

            patches.append(patches_this)
            labels.append(labels_this)
            image_ids.append(image_ids_this)

        patches = np.concatenate(patches, axis=0)
        labels = np.concatenate(labels, axis=0)
        image_ids = np.concatenate(image_ids, axis=0)
        self.patches = patches
        self.labels = labels
        self.image_ids = image_ids
        self.patches_img = np.zeros((self.patches.shape[0],3,2*half_patch_size+1, 2*half_patch_size+1))
        for i, annotfile in enumerate(train_annots):


            imgpath = train_imgs[i]
            img = tifffile.imread(imgpath)
            img = img[:3,:,:]
            patchidx = np.where(self.image_ids==i)[0]
            for p in patchidx:
                patch = crop_patch(img, self.patches[p])
                self.patches_img[p,:,:,:] = patch
        #normalize
        if mean_patch is None:
            mean_patch = np.mean(self.patches_img)
        if std_patch is None:
            std_patch = np.std(self.patches_img)
        self.patches_img = self.patches_img - mean_patch

        self.patches_img = self.patches_img / std_patch
        self.mean_std = (mean_patch, std_patch)
    def __getitem__(self,i):
        img = self.patches_img[i,:,:,:].copy()
        #random noise
        img = img + np.random.randn(*img.shape)*np.random.rand(1)*0.1

        #random data augmentation using scale and bias
        scale = 0.5 + 0.5*np.random.rand(1)
        bias = np.random.rand(1)
        img = img*scale + bias
        label = self.labels[i]
        return torch.Tensor(img), np.float32(label)
    def __len__(self):
        return self.patches_img.shape[0]
def train_model(model, rootdir, train_files,batchsize=64,\
 lr=0.01, momentum=0.9, weight_decay=0.0001, num_epochs=10, mean_patch=277,std_patch=186):

    optimizer = torch.optim.SGD(model.parameters(),lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()



    for epoch in range(num_epochs):
        dataset = FociPatchDataset(rootdir,train_files, mean_patch=mean_patch, std_patch=std_patch, pos_frac=0.4)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,shuffle=True)
        num_classes = int(np.max(dataset.labels)+1)
        total_loss = 0
        acc = 0
        count = 0
        for i, (x,y) in enumerate(data_loader):
            optimizer.zero_grad()
            x_var = Variable(x)
            y_var = Variable(y)
            pred_var = model(x_var)
            loss_val = loss_fn(pred_var.view(pred_var.size(0)),y_var)
            loss_val.backward()
            optimizer.step()
            total_loss = total_loss + loss_val.data[0]
            pred_scores = pred_var.data.numpy()
            pred_scores = pred_scores.reshape(-1)

            pred_labels = pred_scores>0
            acc = acc + np.sum(pred_labels==y.numpy().reshape(-1))
            count = count + y.size(0)
            if i%10 == 0:
                print('Epoch: {:d}, Iter: {:d}, Loss: {:.3f}, Accuracy: {:.3f}'.format(epoch,i,total_loss/(float(i+1)), acc/float(count)))



        #torch.save(model.state_dict(), str(epoch)+'.pt')

    return model

class FociDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, train_files,stride=1, half_patch_size=8, crop_size=512, mean_patch=None,std_patch=None):
        train_imgs = [os.path.join(rootdir, 'imgs', x+'.tif') for x in train_files]
        train_annots = [os.path.join(rootdir, 'annots', x+'.json') for x in train_files]
        self.images = []
        self.labelmaps = []
        self.weights = []
        self.crop_size = crop_size
        self.hps = half_patch_size
        mean = 0
        sqmean = 0

        for i, e in enumerate(train_annots):
            with open(e,'r') as f:
                annot = json.load(f)
            foci = annot['foci']
            pos_locations = []
            for x in foci:
                if [] in x:
                    x.remove([])
                if len(x)==0:
                    continue
                x = np.array(x)
                pos_locations.append(x)
            if len(pos_locations)==0:
                continue
            pos_locations = np.concatenate(pos_locations, axis=0)/stride
            imfile = train_imgs[i]

            img = tifffile.imread(imfile)
            img = img.astype(float)
            img = img[:3,:,:]
            img = img - np.mean(img)
            img = img/np.std(img)
            shape = int(img.shape[1]/stride), int(img.shape[2]/stride)
            labelmap = np.zeros(shape)
            weights = np.ones(shape)
            print(shape)
            y, x = np.unravel_index(np.arange(labelmap.size), labelmap.shape)
            pts = np.concatenate((x.astype(float).reshape((-1,1)),y.astype(float).reshape((-1,1))),axis=1)
            dist = pairwise_distances(pts, pos_locations)
            mindist = np.min(dist, axis=1).reshape(labelmap.shape)
            labelmap[mindist<1]=1
            weights[mindist<5]=0
            weights[mindist<1]=1000

            print(labelmap.shape, labelmap.shape)
            self.images.append(img[np.newaxis,:,:,:].astype(float))
            self.labelmaps.append(labelmap[np.newaxis,:,:])
            self.weights.append(weights[np.newaxis,:,:])

        self.images = np.concatenate(self.images, axis=0)
        self.labelmaps = np.concatenate(self.labelmaps, axis=0)
        self.weights = np.concatenate(self.weights, axis=0)
        print(self.images.shape, self.labelmaps.shape, self.weights.shape)
        #self.labelmaps = self.labelmaps[:,half_patch_size:-half_patch_size, half_patch_size:-half_patch_size]
        mean = np.mean(self.images)
        self.images = self.images - mean
        std = np.std(self.images)
        self.images = self.images/std
        self.mean_std = (mean,std)
        self.ally, self.allx = np.unravel_index(np.arange(2048**2), (2048,2048))

    def __getitem__(self,i):
        notdone=True
        # while(notdone):
        #     starty = 0#np.random.choice(self.images.shape[2]-self.crop_size)
        #     startx = 0#np.random.choice(self.images.shape[3]-self.crop_size)
        #     tmp_labelmap = self.labelmaps[i,starty:(starty+self.crop_size),startx:(startx+self.crop_size)]
        #     if np.sum(tmp_labelmap)==0:
        #         notdone=True
        #     else:
        #         notdone=False


        #img = self.images[i,:,starty:(starty+self.crop_size),startx:(startx+self.crop_size)].copy()
        img = self.images[i,:,:,:].copy()
        #random noise
        img = img + np.random.randn(*img.shape)*np.random.rand(1)*0.1

        #random data augmentation using scale and bias
        scale = 0.5 + 0.5*np.random.rand(1)
        bias = np.random.rand(1)
        img = img*scale + bias

        #add spurious negative foci
        num_spurious = np.random.choice(10)

        for _ in range(num_spurious):
            x = np.random.choice(img.shape[1])
            y = np.random.choice(img.shape[2])
            c = np.random.choice(2)
            w = np.random.rand()*200
            sigma = np.random.rand()*10
            dist = (self.allx-x)**2 + (self.ally-y)**2
            gauss = np.exp(-dist/(2*sigma*sigma))
            img[c,:,:] = img[c,:,:] + w*gauss.reshape((img.shape[1],img.shape[2]))

        img = torch.Tensor(img)
        #labelmap = self.labelmaps[i,starty:(starty+self.crop_size),startx:(startx+self.crop_size)].copy()
        #labelmap = labelmap[self.hps:-self.hps,self.hps:-self.hps]
        labelmap = self.labelmaps[i,:,:]
        labelmap = torch.Tensor(labelmap)
        #weight = self.weights[i,starty:(starty+self.crop_size),startx:(startx+self.crop_size)].copy()
        #weight = weight[self.hps:-self.hps,self.hps:-self.hps]
        weight = self.weights[i,:,:]
        weight = torch.Tensor(weight)
        return img, labelmap,weight

    def __len__(self):
        return self.images.shape[0]


def compute_focal_loss_weights(scores, targets,gamma):
    probs = F.sigmoid(scores)
    oneminuspt = probs*(1-targets) + (1-probs)*targets
    weights = oneminuspt**gamma
    return weights

def train_model_fcn(model, rootdir, train_files,stride=1, batchsize=1, need_sigmoid=False, \
 lr=0.01, momentum=0.9, weight_decay=0.0001, num_epochs=100,checkpointdir='checkpoints', gamma=None):

    optimizer = torch.optim.SGD(model.parameters(),lr, momentum=momentum, weight_decay=weight_decay)

    dataset = FociDataset(rootdir,train_files, stride=stride)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,shuffle=True)

    checkpointdir = os.path.join(rootdir, checkpointdir)
    if not os.path.isdir(checkpointdir):
        os.makedirs(checkpointdir)

    for epoch in range(num_epochs):


        total_loss = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        count = 0
        for i, (x,y,w) in enumerate(data_loader):
            optimizer.zero_grad()
            x_var = Variable(x)
            y_var = Variable(y)
            pred_var = model(x_var)
            weight = w
            if gamma is not None:
                weight = compute_focal_loss_weights(pred_var,y_var, gamma)
                weight = weight*w
            print(x_var.shape, pred_var.shape, y_var.shape, weight.shape)
            if need_sigmoid:
                loss_val = F.binary_cross_entropy_with_logits(pred_var[:,0,:,:],y_var, weight=weight)
            else:
                loss_val = F.binary_cross_entropy(pred_var[:,0,:,:],y_var, weight=weight)
            loss_val.backward()
            optimizer.step()
            total_loss = total_loss + loss_val.data[0]

            pred_scores = pred_var.data.numpy()
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_scores>0
            true_labels = y.numpy().reshape(-1)
            tp = tp + np.sum(pred_labels*true_labels)
            fp = fp + np.sum(pred_labels*(1-true_labels))
            fn = tn + np.sum((1-pred_labels)*true_labels)
            tn = fn + np.sum((1-pred_labels)*(1-true_labels))

            count = count + y.size(0)
            if i%1 == 0:
                prec = tp/(tp+fp)
                rec = tp/(tp+fn)
                print('Epoch: {:d}, Iter: {:d}, Loss: {:.3f}, Precision: {:.3f}, Recall: {:.3f}'.format(epoch,i,total_loss/(float(i+1)), prec,rec))


        thisdir = os.path.join(checkpointdir, str(epoch))

        if not os.path.isdir(thisdir):
            os.makedirs(thisdir)
        torch.save(model.state_dict(), os.path.join(thisdir, 'weights.pt'))


    return model






def apply_model(model,img, mean, std, median_filt=False, synthetic_noise=0):
    img = img.astype(float)
    mean = np.mean(img)
    std = np.std(img)

    print(img.shape)
    imshape = img.shape
    if median_filt:
        img = medfilt(img, 3)
    img = (img - mean)/std
    img = img[np.newaxis, :2,:,:]
    print(img.shape)
    img = img + np.random.randn(*img.shape)*synthetic_noise
    img = Variable(torch.Tensor(img))
    output = model(img)
    scores = output.data.numpy()[0,0,:,:]
    return scores

def nonmax(scores, maxdet):
    y, x = np.unravel_index(np.arange(scores.size),scores.shape)
    points = np.concatenate((x.reshape((-1,1)),y.reshape((-1,1))),axis=1)

    idx = np.argsort(-scores.reshape(-1))
    sorted_scores = -np.sort(-scores.reshape(-1))
    chosen = np.zeros((maxdet,2))
    chosen_scores = np.zeros(maxdet)
    chosen_num = 0
    count = 0
    while(chosen_num<maxdet and count<scores.size):
        point = points[idx[count],:]
        to_choose=True
        if chosen_num>0:
            past_points = chosen[:chosen_num,:]
            diff = past_points - point.reshape((1,2))
            mindist = np.sqrt(np.min(np.sum(diff**2,axis=1)))
            if mindist<5:
                to_choose=False
        if to_choose:
            chosen[chosen_num,:] = point
            chosen_scores[chosen_num] = sorted_scores[count]
            chosen_num = chosen_num+1
        count = count+1
    return chosen, chosen_scores

def nonmax_local(img, nbr=3):
    max_locs = np.zeros((img.shape[1], img.shape[2]), dtype=bool)
    hn = int((nbr-1)/2)
    for i in range(hn,img.shape[1]-hn):
        for j in range(hn,img.shape[2]-hn):
            ismax = img[0,i,j] == np.max(img[0,i-hn:i+hn+1,j-hn:j+hn+1])
            ismax = ismax or (img[1,i,j] == np.max(img[1,i-hn:i+hn+1,j-hn:j+hn+1]))
            max_locs[i,j] = ismax

    return max_locs

def rescore_and_nms_dets(img, scores, maxdet):
    scores=F.sigmoid(torch.Tensor(scores)).numpy()
    scores = resize(scores, (img.shape[1],img.shape[2]), mode='constant', cval=-np.inf)
    max_locs = nonmax_local(img)
    y, x = np.where(max_locs)
    stride = img.shape[1]/scores.shape[1]
    points = np.concatenate((x.reshape((-1,1)),y.reshape((-1,1))),axis=1)
    scores_points = scores[points[:,0], points[:,1]]
    assert(scores_points.shape[0]==points.shape[0])
    idx = np.argsort(-scores_points.reshape(-1))
    sorted_scores = -np.sort(-scores_points.reshape(-1))
    chosen = np.zeros((maxdet,2))
    chosen_scores = np.zeros(maxdet)
    chosen_num = 0
    count = 0
    while(chosen_num<maxdet and count<scores.size):
        point = points[idx[count],:]
        print(point)
        to_choose=True
        if chosen_num>0:
            past_points = chosen[:chosen_num,:]
            diff = past_points - point.reshape((1,2))
            mindist = np.sqrt(np.min(np.sum(diff**2,axis=1)))
            if mindist<5:
                to_choose=False
        if to_choose:
            chosen[chosen_num,:] = point
            chosen_scores[chosen_num] = sorted_scores[count]
            chosen_num = chosen_num+1
        count = count+1
    return chosen, chosen_scores, scores




def save_dets(model, rootdir, files, outdir, mean, std, median_filt=False, maxdet=500, synthetic_noise=0):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    imgs = [os.path.join(rootdir, 'imgs', x+'.tif') for x in files]
    outputs = [os.path.join(outdir, x+'.json') for x in files]
    for i, impath in enumerate(imgs):
        img = tifffile.imread(impath)

        scores = apply_model(model,img, mean, std, median_filt, synthetic_noise=synthetic_noise)
        foci, foci_scores = nonmax(scores, maxdet)
        outputfile=outputs[i]
        outdir_this = os.path.dirname(outputfile)
        if not os.path.isdir(outdir_this):
            os.makedirs(outdir_this)
        with open(outputfile, 'w') as f:
            json.dump(dict(foci=foci.tolist(),foci_scores=foci_scores.tolist()),f)
        print('Done {:d}'.format(i))
