import argparse,os
import matplotlib.pyplot as plt
from DoolCentroid import DoolCentroid

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required=True, type=str,
        help='path to test directory containing IN/ and Reject/ directories')
    args = parser.parse_args()
    argdict = vars(args)

    # instatansiate the class. initial argments IN and BAD are required; 
    # defualt centroid_num=2; changing this is untested December 22
    CENT = DoolCentroid(directory=argdict["directory"])

    # images saved in input_directory stored to CENT.ImageArrays class variable 
    # default image_color=0 is red. image_col=3 to average all RGB (unnessecary)
    CENT.GetImages()

    # estimate background using median filter. estimates stored in CENT.BackgroundArrays
    # background reduced image arrays are stored in CENT.ReducedImageArrays
    # set plot_prompt=True to study each estimated background and resultant reduced images
    CENT.EstimateBackground(filter_size_pixels=40, plot_prompt=False) 

    # detect peaks on the reduced Images and calculate limits for centroid windows across all images
    # default threshold=20. used for peak detection with ndimage NOT for centroid calculations in other class functions
    # default windows have padding=5 pixels. 
    CENT.CreateWindows()
    CENT.DisplayWindows()

    # locate center-of-mass centroids inside each window 
    # timestamped centroids stored to CENT.CentroidDataFrame
    CENT.LocateCentroids()

    print(CENT.CentroidDataFrame)

    CENT.CreateDataCSV()  # save data frame to csv
    CENT.CreateGifs()     # create gif of images with windows
