![Screenshot](/screenshot.png)



# Installation

To download this package, click [here](https://github.com/bharath272/centrosome-analysis/archive/master.zip)

## Python

This tool has been tested with **Anaconda**, **python 3.6.8**.
To install this version of python, install Anaconda by following this [link](https://docs.anaconda.com/anaconda/install/). Follow links for downloading the Python 3.7 version.

## Pytorch
This tool has been tested with **pytorch 0.4.1**
Install this version of pytorch by running:
```
conda install pytorch=0.4.1 -c pytorch
```

## Tifffile
This tool has been tested with **tifffile 0.14.0**
Install this version of tifffile by running:
```
conda install -c conda-forge tifffile
```

# Running
To run this tool, first open the Terminal.
Change into the directory containing this code
```
cd /path/to/centrosome-analysis
```
Then run centosome_analysis_gui.py
```
python centrosome_analysis_gui.py
```

Analysis can be performed through the following steps.
1. First, click on File->Load centrosome detector. Choose the centrosome detector file `foci_model.pt`

2. Second, click on File->Load cell model. Choose the cell segmentation file `cell_model.pt`. These first two steps only need to be done once per session.

3. Then, to analyze an image, lick on File->Open Image and choose the image you want to analyze. The image will be displayed in false color, with the first channel represented as red, second as green, third as blue and fourth as cyan. The tool currently does not allow for more than 4 channels. Individual channels can be visualized by choosing different options in the `Display` drop down menu in the `Visualization` toolbar. You can also zoom into particular regions of the image by clicking the `zoom` button and then identifying a region by clicking aand dragging on the image. Click on `Reset zoom` to return to the original resolution.

4. Next, you will need to choose the channel which represents PCM, the channel which represents Centrin and the channel which represents DAPI in the `Channels` toolbar. PCM and Centrin channels are needed for centrosome detection. Cell segmentation additionally needs DAPI.

5. Next, click on `Run ML models` to run the centrosome detector and the cell segmentation. These models will be run on the image and the results displayed as an overlay on the image. The cell segmentation will be shown as colored boundaries, with the boundaries being thick if the model thinks the cell is centrosome-amplified (more than 4 centrosomes detected). Centrosomes will be depicted either as circles or with a `+` sign (the tool clusters the centrosomes and picks one cluster of centrosomes per cell for analysis. The cluster chosen is the one shown with circles).

6. You can change the threshold for the centrosome detection and cell segmentation. Increasing the threshold for the centrosome detection leads to fewer detections. Increasing the threshold for the cell segmentation leads to more cells.

7. You can also make corrections to the centrosome detections and cell boundaries.
   1. You can make two kinds of corrections to the centrosome detections:
      1. You can add new centrosomes by first clicking the `Add centrosomes` button and then clicking to add centrosomes.
      2. You can remove spurious foci by first clicking the `Remove centrosomes` button and then clicking on the spurious
    centrosomes.

   2. You can make three kinds of corrections to the cell segmentations
      1. You can add cell boundaries that were missed. To do so, click `Add cell boundary` and draw the boundary on the image.
      2. You can merge cells that were wrongly separated. To do so, click `Choose cells to merge`, click on the cells you want to merge and then click `Merge cells`.
      3. You can remove cells from analysis by clicking `Remove cell from analysis` and then clicking `Remove cell from analysis`.
 
   3. You can also change the labeling of cells as amplified or not by clicking `Mark cells as amplified` and `Mark cells as not amplified` and clicking on cells.
   4. Once you have made corrections, you can save these corrections by clicking File->Save corrections.

7. You can now make choices pertaining to the kind of analysis you want to make. Once you have decided the kind of analysis needed and the various parameters of this analysis, click on `Analyze and save` to finish the analysis and save the results. For the first image in a session, you will be prompted for the filename you want to save it to. The filename must be a `csv` file. If the file exists, new results will be appended to the previous results.

