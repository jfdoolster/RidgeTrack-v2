# RidgeTrack-V2                                 

Feb 15 2023 JFD

Software to processes fire-i images ('frames') from input_directory, 
estimate the background, detect beam locations and create "windows",
then calcuate the CoM centroid of the spots contained within each window.

## Usage

Software will run on any computer with the correct python environment (see installation notes below), 
but was designed to run on "Dooley" windows Laptop in IBCA

### **RECOMMENDED**:

Open a powershell terminal and run the commands
``` console
> cd ~/Desktop/RidgeTrack-v2 
> python main.py -d /path/to/directory
```

* [main.py](./main.py) uses [DoolCentroid.py](./DoolCentroid.py). software assumes there is an `IN` directory  
inside `/path/to/directory` where JPG images are stored. 
* JPG images must be in the form `Frame_n_%d_%m_%Y_%H_%M_%S_%f`

### standalone functionality (not recommended):
The same processing is possible using the less object oriented program:

``` console
> cd ~/Desktop/RidgeTrack-v2
> python standalone.py -d /path/to/directory
```

* [standalone.py](./standalone.py) does **NOT** use [DoolCentroid.py](./DoolCentroid.py) but has similar architecture. 
this approach is less efficent and much more prone to bugs as it is updated.

## Summary

centroiding algorithm is largely based on information found at
http://ugastro.berkeley.edu/infrared09/PDF-2009/centroid-error.pdf

The fundamental assumption is that the JPG file names have the form
`Frame_n_%d_%m_%Y_%H_%M_%S_%f.JPG`
In the Fire-i software this is **Frame_$FI_$TS**

### Class Functions Overview

Class funtions are summarized below, more detail is available as in-code comments.

#### DoolCentroid.EstimateBackground():
estimates background for each image using median filter with default size 40pix. 
new background estimates are only created when the current image average changes 
significantly from the previous image to save processing time.

#### DoolCentroid.CreateWindows():
uses thresholding and scipy.ndimage functions to detect beam locations in background-reduced images.
windows take into account all images and grow according to beam motion.

#### DoolCentroid.LocateCentroids():
determines CoM centroid within each window. window is created by cropping background-reduced images.
stores centroid data to DoolCentroid.CentroidDataFrame class variable.

#### DoolCentroid.CreateDataCSV():
saves data to csv file. Care was made to ensure that output csv matched previous csv files created with
the original RidgeTrack program from 2019.

#### DoolCentroid.CreateGifs():
creates gifs for visual inspection. generated gifs are not expected to reveal quantitative data, but serve
as a straight forward method of first approximation centroid validation and can be used for qualitative 
discussions of beam motion.  

## Notes

While beam intensity doesnt change, IBCA lights turing on increaes the background and therefore the resultant 
reduced image beam appears to be much dimmer. CoM centroiding shouldnt be effected too much by this, since windows
are static across images, but may want to play with filter sizes and/or make the size dynamic based on average counts. 

to install/upgrade libraries:

``` console
> python -m pip install --upgrade pip
> python -m pip install --upgrade <package_name>
```
