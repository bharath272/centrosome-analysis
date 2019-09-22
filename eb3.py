#!/Users/divyaganapathisankaran/anaconda/bin/python2.7
import numpy as np
from PIL import Image
from PIL import ImageFilter
from scipy.ndimage.measurements import label
from scipy.signal import convolve2d
import matplotlib
import tifffile
import matplotlib.pyplot as plt
from collections import Counter
import os
import sys
import argparse

def imwrite(filename, image):
    image = (image*255)/np.max(image)

    image = Image.fromarray(image.astype(np.uint8))
    image.save(filename)

def median_filter(eb3):
    eb3 = eb3*255
    eb3 = eb3.astype(np.uint8)
    eb3 = Image.fromarray(eb3).filter(ImageFilter.MedianFilter(3))
    eb3 = np.array(eb3).astype(float)/255.
    return eb3

def box_filter(eb3, radius):
    k = np.ones((radius, radius))/(float(radius**2))
    return convolve2d(eb3, k, 'same')


def get_all_linesegments(eb3, rel_thresh):
    #blur the image
    eb3_blurred = box_filter(eb3,10)

    #look for image locations that are more intense than nearby pixels by rel_thresh
    eb3_sharp = (eb3-eb3_blurred > rel_thresh)

    #identify connected components
    labeled_segments, _ = label(eb3_sharp, np.ones((3,3)))
    return labeled_segments

def preprocess(eb3_raw, rel_thresh):
    eb3 = eb3_raw.astype(float)
    eb3 = eb3/np.max(eb3)
    eb3 = median_filter(eb3)
    labeled_segments = get_all_linesegments(eb3, rel_thresh)
    return eb3, labeled_segments

def label_linesegments_with_distance_count(labeled_segments, center):
    #find image locations on eb3 comets
    indices = np.where(labeled_segments!=0)
    y = indices[0]
    x = indices[1]

    #estimate their distance from center
    pixel_distances = np.sqrt((y-center[1])**2 + (x-center[0])**2)

    #get unique segments
    unique_segments = np.unique(labeled_segments[labeled_segments!=0])

    #run through pixels and record the farthest distance for each
    segment_distances = {x:-1 for x in unique_segments}
    farthest_pixel = {x:(0,0) for x in unique_segments}

    for i in range(y.size):
        label = labeled_segments[y[i],x[i]]
        if pixel_distances[i]>segment_distances[label]:
            segment_distances[label] = pixel_distances[i]
            farthest_pixel[label] = (y[i],x[i])

    #Count the number of pixels in each segment
    segment_counts = Counter(labeled_segments[labeled_segments!=0].tolist())

    return unique_segments, segment_counts, segment_distances, farthest_pixel

def find_foci_around_center(unique_segments, segment_counts, segment_distances, farthest_pixel, length_thresh, radii):
    foci = []
    for i,r in enumerate(radii):
        foci_this = [x for x in unique_segments if segment_counts[x]>length_thresh and segment_distances[x]<r]
        foci.append(foci_this)
    return foci

def analyze_eb3(labeled_segments, eb3, center, radii, length_thresh, abs_thresh, vis_dir=None, centrosome=None, vis=False):
    unique_segments, segment_counts, segment_distances, farthest_pixel = label_linesegments_with_distance_count(labeled_segments, center)
    foci = find_foci_around_center(unique_segments, segment_counts, segment_distances, farthest_pixel, length_thresh, radii)
    counts = [len(x) for x in foci]

    #now look at pixels in the cell body
    cell_body = eb3>abs_thresh


    #Compute distance to center for each pixel
    x,y = np.meshgrid(np.arange(labeled_segments.shape[1]), np.arange(labeled_segments.shape[0]))
    dist = np.sqrt((y-center[1])**2 + (x-center[0])**2)

    #compute cell areas
    areas = []
    for r in radii:
        areas.append(np.sum((dist<r) & cell_body))

    #compute differentials
    diff_counts = []
    diff_areas = []
    diff_densities = []

    for i, radius in enumerate(radii):
        if i==0:
            diff_areas.append(areas[i])
            diff_counts.append(counts[i])
            diff_densities.append(float(counts[i])/float(areas[i]))
        else:
            diff_areas.append(areas[i]-areas[i-1])
            diff_counts.append(counts[i]-counts[i-1])
            diff_densities.append(float(counts[i]-counts[i-1])/float(areas[i]-areas[i-1]))

    if vis:
        if not os.path.isdir(vis_dir):
            os.makedirs(vis_dir)
        for i,r in enumerate(radii):
            cell_area_img = get_cell_area_image(cell_body, dist, r)
            imwrite(os.path.join(vis_dir, 'cell_{:f}.png'.format(r)),cell_area_img)
            labeled_segs_img = get_labeled_segments_image(labeled_segments, centrosome, foci[i], farthest_pixel, dist, r)
            imwrite(os.path.join(vis_dir, 'foci_{:f}.png'.format(r)),labeled_segs_img)
    return diff_counts, diff_areas, diff_densities



def get_cell_area_image(cell_area, dist, radius):
    vis_area = np.zeros((cell_area.shape[0],cell_area.shape[1],3))
    vis_area[:,:,1] = cell_area
    vis_area[np.logical_and(dist<=radius+2, dist>=radius-2),:]=0.5

    return vis_area
def get_labeled_segments_image(labeled_segments, centrosome, foci, farthest_pixel, dist, radius):
    vis_labeled_segs = np.zeros((labeled_segments.shape[0], labeled_segments.shape[1], 3))
    vis_labeled_segs[:,:,1] = (labeled_segments!=0)
    vis_labeled_segs[:,:,0] = (centrosome-np.min(centrosome))/(np.max(centrosome)-np.min(centrosome))
    vis_labeled_segs[np.logical_and(dist<=radius+2, dist>=radius-2),:]=0.5
    points = [farthest_pixel[x] for x in foci]
    boxes = [(x-3,y-3,x+3,y+3) for (y,x) in points]
    for b in boxes:
        xmin, ymin, xmax, ymax = b
        xmin = max(0,xmin)
        ymin = max(0,ymin)
        xmax = min(labeled_segments.shape[1],xmax)
        ymax = min(labeled_segments.shape[0],ymax)
        vis_labeled_segs[ymin:ymax,xmin:xmax,2] = 1
        vis_labeled_segs[ymin:ymax,xmin:xmax,1] = 0
    return vis_labeled_segs
