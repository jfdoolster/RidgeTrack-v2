import matplotlib.pyplot as plt
from DoolCentroid_JFD import Centroid

if __name__ =="__main__":
    #testdir = 'C:/Users/dooley/Documents/brsSpot/'
    testdir = "/home/dooley/Desktop/DOOLEY-20221211/20221103-DOOLEY"
    IN = testdir + "/IN/" 
    BAD = testdir + "/Reject/" 
    OUT = testdir + ""  # <-- add filename in quotes if desired

    CENT = Centroid(input_directory = IN, reject_directory=BAD)
    CENT.GetImages()
    CENT.EstimateBackground()
    CENT.CreateWindows()
    CENT.LocateCentroids()
    CENT.CreateDataCSV(csv_path = OUT)

    CENT.CreateGifs(gif_path = OUT)




    
    
