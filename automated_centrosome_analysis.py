import numpy as np
import centrosome_analysis_backend
import argparse
import json
import os
import tifffile




def do_analysis_single(filename, channelorder, results_file, foci_model, ml_cell_seg_model, mean, std, radii, bgradii, det_thresh, cell_thresh):
    print(filename)

    detection_file = os.path.splitext(filename)[0]+'.json'
    cell_probability_file = os.path.splitext(filename)[0] + '.npy'


    img = tifffile.imread(filename)
    print(channelorder)
    img = img[channelorder,:,:]
    foci, foci_scores = centrosome_analysis_backend.run_detection_model(img, foci_model, mean, std)
    with open(detection_file, 'w') as f:
        json.dump(dict(foci=foci.tolist(),foci_scores=foci_scores.tolist()), f)


    cell_probabilities, cell_bmap = centrosome_analysis_backend.run_cell_model(img, ml_cell_seg_model, mean, std)
    np.savez(cell_probability_file, cell_probs=cell_probabilities, cell_bmap = cell_bmap)
    print(np.sum(foci_scores>det_thresh))
    if np.sum(foci_scores>det_thresh)==0:
        return
    detections = foci[foci_scores>det_thresh,:]

    detections = np.round(detections).astype(int)
    cell_labels, cell_map = centrosome_analysis_backend.get_cell_labels(cell_bmap, cell_probabilities, detections, cell_thresh)
    amplified, chosen_for_analysis, unique_labels = centrosome_analysis_backend.cell_analysis(detections, cell_labels)
    intensities, areas, densities = centrosome_analysis_backend.intensity_profile(img[0,:,:], detections, cell_labels, chosen_for_analysis, cell_map,radii, bgradii)
    centrosome_analysis_backend.save_to_csv(filename, results_file, intensities, areas, densities, amplified, radii)


def parse_args():
    parser = argparse.ArgumentParser('Parser')
    parser.add_argument('--cell_model', default='cell_model.pt')
    parser.add_argument('--foci_model', default='foci_model.pt')
    parser.add_argument('--mean', default=118, type=float)
    parser.add_argument('--std',default=23,type=float)
    parser.add_argument('--det_thresh', default=0.7, type=float)
    parser.add_argument('--cell_thresh', default=0.2, type=float)
    parser.add_argument('--pixel_scale', default=0.13, type=float)
    parser.add_argument('--min_radius', default=0.13, type=float)
    parser.add_argument('--max_radius', default=5.33, type=float)
    parser.add_argument('--step', default=0.13, type=float)
    parser.add_argument('--input_set', default=None, type=str)
    parser.add_argument('--results_file', default='results.csv', type=str)
    return parser.parse_args()

def read_input_set_file(input_set):
    with open(input_set, 'r') as f:
        lines = f.readlines()
    print(lines[0])
    lines = [x.strip().split(',') for x in lines]

    header = lines[0]
    print(header)
    filename_idx = header.index('Filename')
    pericentriolar_idx = header.index('Pericentriolar')
    centrin_idx = header.index('Centrin')
    dapi_idx = header.index('DAPI')
    scale_idx = header.index('Microns/pixel')
    filenames = [x[filename_idx] for x in lines[1:]]
    pericentriolar = [int(x[pericentriolar_idx])-1 for x in lines[1:]]
    print(pericentriolar)
    centrin = [int(x[centrin_idx])-1 for x in lines[1:]]
    dapi = [int(x[dapi_idx])-1 for x in lines[1:]]
    scales = [float(x[scale_idx]) for x in lines[1:]]
    channelorder = [x for x in zip(pericentriolar, centrin, dapi)]
    return filenames, channelorder, scales

if __name__=='__main__':
    params = parse_args()
    foci_model = centrosome_analysis_backend.load_foci_model(params.foci_model)
    cell_model = centrosome_analysis_backend.load_cell_model(params.cell_model)
    filenames, channelorders, scales = read_input_set_file(params.input_set)
    radii = np.arange(params.min_radius, params.max_radius, params.step)
    bgradii = np.array([7,8,9])
    for i in range(len(filenames)):

        filename = filenames[i]
        print('Analyzing {}'.format(filename))
        channelorder = list(channelorders[i])
        do_analysis_single(filename, channelorder, params.results_file, foci_model, cell_model, params.mean, params.std, radii/scales[i], bgradii/scales[i], params.det_thresh, params.cell_thresh)
