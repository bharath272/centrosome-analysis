import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.ndimage.measurements import label
import torch
import ml_foci_detect
import ml_cell_segmentation
import os
import cell_segmentation
import eb3 as eb3_analysis
def load_foci_model(foci_model_file):

    model = ml_foci_detect.MultiChannelCombinedScorer()
    x = torch.load(foci_model_file)
    model.load_state_dict(x)
    return model

def load_cell_model(cell_model_file):

    model = ml_cell_segmentation.CellSegmentationModel(use_scale_estimator=False)
    x = torch.load(cell_model_file)

    model.load_state_dict(x)
    return model

def run_detection_model(img, foci_model, mean, std):
    img = img.astype(float)
    scores = ml_foci_detect.apply_model(foci_model, img, mean, std)
    foci, foci_scores = ml_foci_detect.nonmax(scores, 500)
    foci = foci.astype(int)
    return foci, foci_scores
def run_cell_model(img, ml_cell_segmentation_model, mean, std):
    #the cell model seems to be sensitive to absolute brightness and contrast of the image

    img = img.astype(float)
    mean = np.mean(img)
    std = np.std(img)/10
    cell_prob = ml_cell_segmentation.apply_model(ml_cell_segmentation_model,img, mean, std)
    cell_bmap = cell_segmentation.compute_cell_bmap(cell_prob)
    return cell_prob, cell_bmap




def get_cell_map(cell_probabilities, bmap, boundary_thresh):
    cellmap = cell_segmentation.get_cells(cell_probabilities, bmap, 1-boundary_thresh)
    return cellmap

def get_cell_labels(cellmap, detections):
    #bw = cell_probabilities>cell_probability_thresh

    detections = np.round(detections).astype(int)
    cell_labels = cellmap[detections[:,1],detections[:,0]]

    return cell_labels
def cluster_foci(foci):
    z = linkage(foci)
    T = fcluster(z, 40, criterion='distance')
    return T

def get_avg_background(cell_bmap, img, centroid, bgradii):
    y, x = np.where(cell_bmap)
    centroid = centroid.reshape(-1)
    dist = np.sqrt((x-centroid[0])**2 + (y-centroid[1])**2)
    avgs = []
    for i in range(len(bgradii)-1):
        idx = (dist>bgradii[i]) & (dist<=bgradii[i+1])
        avgs.append(np.mean(img[y[idx],x[idx]]))
    avg = np.mean(avgs)

    return avg

def cell_analysis(detections, labels):



    amplified = []
    chosen_for_analysis = np.zeros(detections.shape[0], dtype=bool)
    unique_labels = np.unique(labels)
    for l in unique_labels:

        this_cell = np.where(labels==l)[0]
        foci = detections[this_cell,:]



        amplified.append(int(foci.shape[0]>4))

        if len(this_cell)==1:
            chosen_for_analysis[this_cell[0]] = True
            continue
        T = cluster_foci(foci)
        counts = np.zeros(int(np.max(np.unique(T)))+1)

        np.add.at(counts, T, 1)
        idx = np.argmax(counts)

        chosen_for_analysis[this_cell[T==idx]]=True
    return amplified, chosen_for_analysis, unique_labels

def intensity_profile(img, detections, labels, chosen_for_analysis, cell_map,radii, bgradii):
    unique_labels = np.unique(labels)

    y, x = np.unravel_index(np.arange(img.size), img.shape)
    y = y.reshape(img.shape)
    x = x.reshape(img.shape)
    intensities = []
    areas = []
    densities = []
    for l in unique_labels:
        foci = detections[(labels==l) & chosen_for_analysis,:]
        centroid = np.mean(foci, axis=0)
        cell_bmap = cell_map==l
        bgscore = get_avg_background(cell_bmap, img, centroid, bgradii)
        if np.isnan(bgscore):
            continue
        img_tmp = img - bgscore


        dist = np.sqrt((y-centroid[1])**2 + (x-centroid[0])**2)




        I = []
        A = []

        for r in radii:
            total_intensity = np.sum(img_tmp[dist<r])
            I.append(total_intensity)
            total_area = np.sum(dist<r)
            A.append(total_area)

        I_o = [I[0]]
        A_o = [A[0]]
        D_o = [I[0]/A[0]]
        for i in range(1, len(radii)):
            I_o.append(I[i]-I[i-1])
            A_o.append(A[i]-A[i-1])
            D_o.append((I[i]-I[i-1])/(A[i]-A[i-1]))

        intensities.append(I_o)
        areas.append(A_o)
        densities.append(D_o)

    return intensities, areas, densities

def preprocess_ebr(eb3, rel_thresh):
    eb3, labeled_segments = eb3_analysis.preprocess(eb3, rel_thresh)
    return eb3, labeled_segments


def eb3_count_density(labeled_segments, eb3, detections, labels, chosen_for_analysis,
    radii, length_thresh, abs_thresh, vis_dir=None, centrosome=None, vis=False):
    diff_counts_all = []
    diff_areas_all = []
    diff_densities_all = []

    unique_labels = np.unique(labels)

    for l in unique_labels:
        if vis_dir is not None:
            vis_dir_this = os.path.join(vis_dir, str(l))
        foci = detections[(labels==l) & chosen_for_analysis,:]
        centroid = np.mean(foci, axis=0)
        d_c,d_a,d_d = eb3_analysis.analyze_eb3(labeled_segments, eb3, centroid, radii, length_thresh, abs_thresh, vis_dir_this, centrosome, vis)
        diff_counts_all.append(d_c)
        diff_areas_all.append(d_a)
        diff_densities_all.append(d_d)
    return diff_counts_all, diff_areas_all, diff_densities_all


def save_to_csv(open_file, save_file, intensities, areas, densities, amplified, radii):
    save_areas_file = os.path.splitext(save_file)[0]+'-areas.csv'
    save_int_file = os.path.splitext(save_file)[0]+'-intensities.csv'

    common_lines = [[open_file, str(i), str(a)] for i,a in enumerate(amplified) ]
    intensity_lines = []
    area_lines = []
    density_lines = []
    intensity_lines = [','.join(common_lines[i] + [str(y) for y in x]) +'\n' for i,x in enumerate(intensities)]

    area_lines = [','.join(common_lines[i] + [str(y) for y in x])+'\n' for i,x in enumerate(areas)]
    density_lines = [','.join(common_lines[i] + [str(y) for y in x]) + '\n' for i,x in enumerate(densities)]
    radii = [str(r) for r in radii]
    headers = ['Filename', 'Cell id', 'Amplified?'] + radii
    headers = ','.join(headers) + '\n'
    if not os.path.isfile(save_file):

        with open(save_file, 'w') as f:
            f.writelines([headers])
        with open(save_areas_file, 'w') as f:
            f.writelines([headers])
        with open(save_int_file, 'w') as f:
            f.writelines([headers])


    with open(save_file, 'a') as f:
        f.writelines(density_lines)
    with open(save_areas_file, 'a') as f:
        f.writelines(area_lines)
    with open(save_int_file, 'a') as f:
        f.writelines(intensity_lines)


def save_eb3_to_csv(open_file, save_file, counts, areas, densities, amplified, radii):
    save_areas_file = os.path.splitext(save_file)[0]+'-areas.csv'
    save_int_file = os.path.splitext(save_file)[0]+'-counts.csv'

    common_lines = [[open_file, str(i), str(a)] for i,a in enumerate(amplified) ]
    count_lines = []
    area_lines = []
    density_lines = []
    count_lines = [','.join(common_lines[i] + [str(y) for y in x]) +'\n' for i,x in enumerate(counts)]

    area_lines = [','.join(common_lines[i] + [str(y) for y in x])+'\n' for i,x in enumerate(areas)]
    density_lines = [','.join(common_lines[i] + [str(y) for y in x]) + '\n' for i,x in enumerate(densities)]
    radii = [str(r) for r in radii]
    headers = ['Filename', 'Cell id', 'Amplified?'] + radii
    headers = ','.join(headers) + '\n'
    if not os.path.isfile(save_file):

        with open(save_file, 'w') as f:
            f.writelines([headers])
        with open(save_areas_file, 'w') as f:
            f.writelines([headers])
        with open(save_int_file, 'w') as f:
            f.writelines([headers])


    with open(save_file, 'a') as f:
        f.writelines(density_lines)
    with open(save_areas_file, 'a') as f:
        f.writelines(area_lines)
    with open(save_int_file, 'a') as f:
        f.writelines(count_lines)
