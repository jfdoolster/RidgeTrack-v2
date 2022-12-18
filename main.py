import matplotlib.pyplot as plt
from DoolCentroid import DoolCentroid

if __name__ =="__main__":
    #testdir = "C:/Users/dooley/Documents/brsSpot/stability/20220927-testing/"
    testdir = "C:/Users/jdooley/Desktop/DOOLEY-20220927/"
    IN = testdir + "/IN/" 
    BAD = testdir + "/Reject/" 
    OUT = testdir + "/"  # <-- add filename in quotes if desired

    # instatansiate the class. initial argments IN and BAD are required; 
    # defualt centroid_num=2; changing this is untested December 22
    CENT = DoolCentroid(input_directory = IN, reject_directory=BAD)

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

    CENT.CreateDataCSV(out_path = OUT)  # save data frame to csv
    CENT.CreateGifs(out_path = OUT)     # create gif of images with windows
