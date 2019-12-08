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
from PIL import Image, ImageDraw
import scale_estimator
from skimage.transform import resize



class CellSegmentationModel(nn.Module):
    def __init__(self,input_channels=3, hidden_channels=10, ksize=5, use_scale_estimator=True):
        super(CellSegmentationModel, self).__init__()
        p = int((ksize-1)/2)
        print(p)
        self.use_scale_estimator = use_scale_estimator
        if self.use_scale_estimator:
            self.scale_estimator = scale_estimator.ScaleEstimator()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels,ksize,stride=2, padding=p)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, ksize, stride=2, padding=p)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, ksize, stride=2, padding=p)
        self.conv4 = nn.Conv2d(hidden_channels, 1, ksize, stride=2, padding=p)
        # self.conv5 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, stride=2, padding=0)
        # self.conv6 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, stride=2, padding=0)
        #self.conv7 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=4, padding=0)
        #self.conv8 = nn.ConvTranspose2d(hidden_channels, 1, 4, stride=4, padding=0)
        self.relu = nn.ReLU()


    def forward(self, x):
        if self.use_scale_estimator:
            scale, bias = self.scale_estimator(x)
            x = x*scale + bias
        output = self.relu(self.conv1(x))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        output = self.conv4(output)


        #output = self.relu(self.conv5(output))
        #output = self.relu(self.conv6(output))
        #output = self.relu(self.conv7(output))
        #output = self.conv8(output)

        return output


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

class CenteredDataset(torch.utils.data.Dataset):
    def __init__(self, rootdir, train_files, imsize):
        super(CenteredDataset, self).__init__()
        self.imsize = imsize
        train_imgs = [os.path.join(rootdir, 'imgs', x+'.tif') for x in train_files]
        train_annots = [os.path.join(rootdir, 'annots', x+'.json') for x in train_files]
        self.images = []
        self.labelmaps = []
        self.foci = []
        labels = []
        image_ids = []
        cell_ids = []
        foci_ids = []
        for i, e in enumerate(train_annots):
            with open(e,'r') as f:
                annot = json.load(f)
            img = tifffile.imread(train_imgs[i])
            label_img = np.zeros((img.shape[1],img.shape[2]))

            foci = annot['foci']
            array_foci = []
            for j,x in enumerate(foci):
                if [] in x:
                    x.remove([])
                if len(x)==0:
                    foci.remove(x)
                    continue
                x = np.array(x)
                x = np.round(x).astype(int)
                array_foci.append(x)
                label_img[x[:,1],x[:,0]] = j+1
                image_ids.append(i*np.ones(x.shape[0]))
                cell_ids.append((j+1)*np.ones(x.shape[0]))
                foci_ids.append(np.arange(x.shape[0]))
            self.images.append(img[np.newaxis,:,:,:])
            self.labelmaps.append(label_img[np.newaxis, np.newaxis,:,:])
            self.foci.append(array_foci)
        self.image_ids = np.concatenate(image_ids).astype(int)
        self.cell_ids = np.concatenate(cell_ids).astype(int)
        self.foci_ids = np.concatenate(foci_ids).astype(int)
        self.images = np.concatenate(self.images, axis=0)
        self.labelmaps = np.concatenate(self.labelmaps, axis=0)
        self.mean = np.mean(self.images)
        self.std = np.std(self.images)


    def __getitem__(self,i):
        image_id = self.image_ids[i]
        cell_id = self.cell_ids[i]
        foci_id = self.foci_ids[i]
        img = self.images[image_id,:,:,:]
        labelmap = self.labelmaps[image_id,:,:,:]==cell_id
        foci = self.labelmaps[image_id,:,:,:]>0
        img = np.concatenate((img, foci),axis=0)
        center = self.foci[image_id][cell_id-1][foci_id,:]
        rx = -5 + np.random.rand()*10
        ry = -5 + np.random.rand()*10
        print(rx,ry)
        box = [center[0] - self.imsize/2 +rx, \
            center[1] - self.imsize/2 + ry, \
            center[0] + self.imsize/2 + rx, \
            center[1] + self.imsize/2 + ry]
        patch = crop_patch(img, box)
        patch = patch - self.mean
        patch = patch/self.std
        labelmap = crop_patch(labelmap, box)
        foci = crop_patch(foci, box)

        return torch.Tensor(patch), torch.Tensor(labelmap[0,:,:].astype(float)), torch.Tensor(100000*foci[0,:,:].astype(float))

    def __len__(self):
        return self.images.shape[0]


def smoothness_loss(output):
    loss_x = torch.mean((output[:,:-1,:] - output[:,1:,:])**2)
    loss_y = torch.mean((output[:,:,:-1] - output[:,:,1:])**2)
    return loss_x + loss_y


def train_model_fcn(model, rootdir, train_files,imsize=256,batchsize=1,\
 lr=0.01, momentum=0.9, weight_decay=0.0001, num_epochs=100,checkpointdir='checkpoints', gamma=None):

    optimizer = torch.optim.SGD(model.parameters(),lr, momentum=momentum, weight_decay=weight_decay)

    dataset = CenteredDataset(rootdir,train_files, imsize)
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
            loss_val = F.binary_cross_entropy_with_logits(pred_var[:,0,:,:],y_var, weight=weight)
            loss_val = 2*loss_val + 200*smoothness_loss(pred_var[:,0,:,:])
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



class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, imfiles, imgdir, annotdir):
        super(SimpleDataset, self).__init__()
        self.images = []
        self.labels = []
        for i, f in enumerate(imfiles):
            imfile = os.path.join(imgdir, f+'.tif')
            maskfile = os.path.join(annotdir, f + '.png')
            if not os.path.isfile(maskfile):
                continue
            img = tifffile.imread(imfile)
            img = img.astype(float)
            img = img - np.mean(img)
            img = img/np.std(img)
            mask = Image.open(maskfile).convert('RGB')
            mask = np.array(mask)
            mask = mask[:,:,0]
            mask = mask.astype(float)/255.
            self.labels.append(mask[np.newaxis, :,:])
            self.images.append(img[np.newaxis,:,:,:])
        self.labels = np.concatenate(self.labels, axis=0)
        self.images = np.concatenate(self.images, axis=0)
        #self.mean = np.mean(self.images)
        #self.std = np.std(self.images)
        #self.images = self.images - self.mean
        #self.images = self.images / self.std

    def __getitem__(self, i):
        return torch.Tensor(self.images[i,:,:,:]), torch.Tensor(self.labels[i,:,:])

    def __len__(self):
        return self.images.shape[0]


def train_model_fcn2(model, imgdir, annotdir, train_files,imsize=256,batchsize=1,\
 lr=0.01, momentum=0.9, weight_decay=0.0001, num_epochs=100,checkpointdir='checkpoints_fcn', gamma=None):

    print(checkpointdir)
    rootdir = os.path.dirname(imgdir)
    optimizer = torch.optim.SGD(model.parameters(),lr, momentum=momentum, weight_decay=weight_decay)

    dataset = SimpleDataset(train_files, imgdir, annotdir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,shuffle=True)

    #checkpointdir = os.path.join(rootdir, checkpointdir)
    print(checkpointdir)

    if not os.path.isdir(checkpointdir):
        os.makedirs(checkpointdir)

    for epoch in range(num_epochs):


        total_loss = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        count = 0
        for i, (x,y) in enumerate(data_loader):
            optimizer.zero_grad()
            x_var = Variable(x)
            y_var = Variable(y)
            pred_var = model(x_var)
            print(x_var.size(), y_var.size(), pred_var.size())
            loss_val = F.binary_cross_entropy_with_logits(pred_var[:,0,:,:],y_var[:,::16,::16])
            loss_val.backward()
            optimizer.step()
            total_loss = total_loss + loss_val.data[0]
            pred_scores = pred_var.data.numpy()
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_scores>0

            true_labels = y.numpy()[:,::16,::16].reshape(-1)
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


def save_annotated_masks(imfiles, imgdir, annotdir):
    for i, f in enumerate(imfiles):
        imfile = os.path.join(imgdir, f+'.tif')
        annotfile = os.path.join(annotdir, f+'.json')
        if not os.path.isfile(annotfile):
            continue
        with open(annotfile, 'r') as a:
            annot = json.load(a)
        img = Image.new('L', (2048,2048), 0)
        for p in annot:
            polygon = [tuple(x) for x in p]
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=255)
        img.save(os.path.join(annotdir, f+'.png'))
        print(i)

def apply_model(model,img, mean, std, median_filt=False, synthetic_noise=0):
    img = img.astype(float)
    imshape = img.shape
    if median_filt:
        img = medfilt(img, 3)
    mean = np.mean(img)
    std=np.std(img)
    img = (img - mean)/std
    img = img[np.newaxis, :3,:,:]
    img = img + np.random.randn(*img.shape)*synthetic_noise
    img = Variable(torch.Tensor(img))
    output = model(img)
    output = torch.sigmoid(output)
    scores = output.data.numpy()[0,0,:,:]
    scores = resize(scores, imshape[1:])

    return scores
