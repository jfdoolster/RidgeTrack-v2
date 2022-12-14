import shutil
import os, glob
import platform
import numpy as np
import pandas as pd
import datetime as dt
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


#testdir = "/home/dooley/Desktop/DOOLEY-20221211/20221103-DOOLEY"
testdir = "C:/Users/jdooley/Desktop/20221103-DOOLEY"
input_directory = testdir + "/IN/" 
reject_directory  = testdir + "/Reject/" 
OUT = testdir + ""  # <-- add filename in quotes if desired

centroid_num = 2

# set input paths as class variables
IN = input_directory
Reject = reject_directory 
centroid_num = centroid_num

# initialize class variables for sharing between class functions
ImagePaths = []         # list of image path names
ImageDates = []         # list of image timestamps (from path names)
ImageArrays = []        # list of 2D arrays for each images (default: red) 
BackgroundArrays = []   # list of 2D background arrays (estimated)
ReducedImageArrays = [] # list of 2D reduced images arrays (ImageArrays[i] - BackgroundArrays[i])

CentroidWindows = []    # pixel window locations for centroiding ([x0, y0], [x1, y1])
CentroidDataFrame = pd.DataFrame() # pandas dataframe for streamlined data viewing 

####
# Utility class functions:
####

def centroid(I): # calculate centroid and error (<x> and sigma_<x>)
	# get full shape of the rectangular peak
	# Note these shapes will have come from the find objects subroutine
	h,w = np.shape(I)

	# create evenly spaced vector for x and y (coordinate axis)
	x   = np.arange(0,w)
	y   = np.arange(0,h)

	# get coordinate matrix from x and y
	X,Y = np.meshgrid(x,y)

	# http://ugastro.berkeley.edu/infrared09/PDF-2009/centroid-error.pdf

	# calculate centroid from intensities, I, and position, X or Y
	cx  = np.sum(X*I)/np.sum(I)
	cy  = np.sum(Y*I)/np.sum(I)
	# calculate error on centroid
	sigx=np.sqrt(np.sum(I*(X-cx)**2)/np.sum(I)**2)
	sigy=np.sqrt(np.sum(I*(Y-cy)**2)/np.sum(I)**2)

	return cx,cy,sigx,sigy

def getFileDate(filename):
	NAME = filename.split('_') # split filename string
	# create date string 
	datestr = '{:s}-{:s}-{:s} {:s}:{:s}:{:s}'.format(NAME[3],NAME[2],NAME[4],NAME[5],NAME[6],NAME[7])
	# date string to datetime value
	return dt.datetime.strptime(datestr,'%m-%d-%Y %H:%M:%S')

def DisplayWindows():

	# display information about each of the windows
	for win_num in range(len(CentroidWindows)):
		x_start = CentroidWindows[win_num][0][0]
		y_start = CentroidWindows[win_num][0][1]
		x_end = CentroidWindows[win_num][1][0]
		y_end = CentroidWindows[win_num][1][1]
		x_size = x_end - x_start
		y_size = y_end - y_start
		print("window_%d: (x0,y0) = (%d, %d), (x1,y1) = (%d, %d), (w,h) = (%d, %d)" % ((win_num+1), x_start, y_start, x_end, y_end, x_size, y_size))

def BackgroundReductionPlot(idx):
	fig, ax = plt.subplots(1,3, sharex=True, sharey=True)

	ax[0].imshow(ImageArrays[idx], vmax=255)
	ax[1].imshow(BackgroundArrays[idx], vmax=255)
	ax[2].imshow(ReducedImageArrays[idx], vmax=255)

	ax[0].set_title("Original")
	ax[1].set_title("Background")
	ax[2].set_title("Reduced Image")
	plt.show()

def SingleFramePlot(idx):
	fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
	ax.imshow(ReducedImageArrays[idx], vmax=255)

	for win_num in range(centroid_num):
		ax.plot(CentroidDataFrame["centroid_x%d" % (win_num+1)][idx], CentroidDataFrame["centroid_y%d" % (win_num+1)][idx],'r.')

if __name__=="__main__":
		
		# first check if input paths exsist and are directories
		if not os.path.exists(input_directory) or not os.path.isdir(input_directory):
			print("`input_directory` path does not exist or is not a directory (%s)" % input_directory)
			exit(1)
		if not os.path.exists(reject_directory) or not os.path.isdir(reject_directory):
			print("`reject_directory` path does not exist or is not a directory" % reject_directory)
			exit(1)

		'''
		Populate ImagePaths, ImageDates, and ImageArrays class variables. 
		Must run this class function first as other class functions require these class variables 
		'''
		# .jpgs are split into red/green/blue. Fire-i is monochrome so red=blue=green
		image_color = 0
		if image_color not in range(4):
			print("image color must be 0 (red), 1 (blue), 2 (green), or 3 (averaged)")
			exit(1)

		# get sorted list of filenames from 'IN' directory
		if platform.system() == 'Windows':
			ImagePaths = sorted(glob.glob(IN + "*.JPG"),key=os.path.getctime)
		else:
			ImagePaths = sorted(glob.glob(IN + "*.JPG"),key=os.path.getmtime)
		print("%d frames found in %s" % (len(ImagePaths), IN))

		InitNumberImages = len(ImagePaths)

		for fname in ImagePaths:
			image_date  = getFileDate(fname) # get date from filename 

			# get single color from jpg
			# Fire-i is monochrome so red=blue=green
			jpg_image = plt.imread(fname,'F')

			image_array = jpg_image[:,:,image_color].astype('float64')

			if image_color == 3:
				for i in [1, 2]:
					image_array += jpg_image[:,:,i].astype('float64')
				image_array = image_array/3


			# append to image array and date to class variables
			ImageArrays.append(image_array)
			ImageDates.append(image_date)	

		'''
		Estimate background for images using large median filter.
		Populate BackgroundArrays and ReducedImageArrays class variables.
		Must first run GetImages() class function	
		'''

		# initialize background and reduced image arrays with the same shape as the image arrays
		image_size = ImageArrays[0].shape
		bg_array = np.zeros(image_size)
		reduced_image_array = np.zeros(image_size)

		prev_mean = None
		prev_std = None

		for i in range(len(ImageArrays)):
			new_background = False  # boolean date for estimating new background (reset each loop)
			show_plots = False      # boolean gate for plotting image reduction steps (reset each loop)

			image_array = ImageArrays[i]
			image_name_sm = ImagePaths[i].split("/")[-1] # file name without directory prefix 

			# simple stats for each image array
			mean = np.mean(image_array)
			std = np.std(image_array)

			if prev_mean == None:     # true only for first loop. 
				new_background = True # must estimate initial background.

			elif (mean < prev_mean-prev_std) or (mean > prev_mean+prev_std): 
				new_background = True # changing image background level requires new estimate
				'''
				# possible improvement:
				change window size for higher background? (lights on, background > 50)
				if mean > 50
					filter_size_pixel = different 
				'''

			# set true to view background plots
			plot_prompt = False 
			filter_size_pixels=40
			# only estimate new background when needed (see above) to save processing time
			if new_background:
				print("(%d of %d): \033[1mEstimating background\033[0m of %s (size=%d px)..." % (i+1, len(ImageArrays), image_name_sm, filter_size_pixels))

				# estimate background array with large median filter (>= twice centroid diameter of ~20px)
				bg_array = ndimage.median_filter(image_array, size=filter_size_pixels)

				if plot_prompt:
					# user input to view plots (plots called at end of current class function)
					res = input("Done! Plot Estimated Background? y/[n] ")
					if res == "y" or res == "Y":
						show_plots = True

			
			print("(%d of %d): Removing Background from %s..." % (i+1, len(ImageArrays), image_name_sm,))

			# subtract background from image array. small median filter (3px) to remove spurious pixels 
			reduced_image_array = ndimage.median_filter((image_array - bg_array), size=3)

			# populate class variables
			BackgroundArrays.append(bg_array)
			ReducedImageArrays.append(reduced_image_array)

			# set simple stats for comparison in next loop
			prev_mean = mean
			prev_std = std

			if show_plots:
				# plot image, estimated background, and reduced image for verification
				BackgroundReductionPlot(i)  

				# user input to continue after viewing plots
				res = input("Continue? [y]/n ")
				if res == "n" or res == "N":
					exit(0)

		'''
		Create subwindows around centroids for unbiased centroiding. 	
		Image is thresholded and 'features' are found using scipy.ndimage functions
		Populates CentroidWindow class function.
		Must first run GetImages() and EstimateBackground() class functions
		'''

		threshold=20 #counts
		# ensure window lists are empty before continuing
		CentroidWindows = [] 
		peak_windows = []

		for i in range(len(ReducedImageArrays)):
			reduced_image_array = ReducedImageArrays[i]

			# all values below threshold (default=20 counts) set to zero
			threshold_array = reduced_image_array
			threshold_array[threshold_array < threshold] = 0

			# label and number the features
			lbl,num = ndimage.label(threshold_array)

			# ensure that the correct number of features are detected. 
			if num != centroid_num:
				# remove image from future processing, copy original file to 'Reject' direcotry, and alert user
				msg = "incorrect number of detected peaks (%d)" % num
				image_name_sm = ImagePaths[i].split("/")[-1]
				shutil.copy2(ImagePaths[i], Reject)
				ImagePaths.pop(i)
				ImageDates.pop(i) 
				ImageArrays.pop(i)
				BackgroundArrays.pop(i)
				ReducedImageArrays.pop(i)
				if type(msg) == str:
					print("%s removed from analysis: %s" % (image_name_sm, msg))
				else:
					print("%s removed from analysis" % image_name_sm)
				continue

			# only if correct num of features:
			# find objects that remain after thresholding
			peaks = ndimage.find_objects(lbl)

			for p in peaks:
				dy, dx = p

				# determine window size for current feature 
				x0,y0 = dx.start, dy.start
				h,w = np.shape(ReducedImageArrays[i][p])
				x1,y1  = x0+w, y0+h
				
				# initial window size added to list
				# note that this
				if i == 0:
					peak_windows.append([(x0,y0),(x1,y1)])
					continue
				
				# sanity check, previous conditional should have removed images with incorrect number of features
				if len(peak_windows) != centroid_num:
					print("Error: Too many windows!?! (should not happen)")
					exit(1)
				
				# feature detection is not nessecarily consistant across images.
				# the following loop ensures that feature1 and feature2 are not swapped by matching current x0 with closest previous x0
				# if centroids are directly above one another (vertical line intercepts both on image) then change 'x0' variables to 'y0'
				min_x_diff, idx = 999, None
				for w in range(len(peak_windows)):
					prev_x0 = peak_windows[w][0][0]
					dif = abs(x0 - prev_x0)
					if dif < min_x_diff:
						min_x_diff, idx = dif, w
				
				# determine minimum x0,x0 and maximum x1,y1 so detected features accross all images are included in the each window area
				win_x0 = min(x0, peak_windows[idx][0][0])
				win_y0 = min(y0, peak_windows[idx][0][1])
				win_x1 = max(x1, peak_windows[idx][1][0])
				win_y1 = max(y1, peak_windows[idx][1][1])

				# set window values to be used during the next loop
				peak_windows[idx] = [(win_x0, win_y0),(win_x1, win_y1)]

		# sanity check (again)
		if len(peak_windows) != centroid_num:
			print("Error: Too many windows!?! (should not happen)")
			exit(1)

		padding = 5 # pix
		for win_num in range(len(peak_windows)):
			(x0,y0) = peak_windows[win_num][0]
			(x1,y1) = peak_windows[win_num][1]
			peak_windows[win_num][0] = (x0-padding, y0-padding)
			peak_windows[win_num][1] = (x1+padding, y1+padding)


		# populate class variable for future class fucntions
		CentroidWindows = peak_windows

		'''
		Locate center-of-mass centroids using windows calculated in CreateWindows class function.
		Populate pandas dataframe ('CentroidDataFrame') class variable.
		Must first run CreateWindows() class function
		'''

		# initialize empty dataframe
		df = pd.DataFrame()

		# add timestamps column to dataframe
		df["timestamp"] = ImageDates

		for win_num in range(len(CentroidWindows)):
			(x0,y0) = CentroidWindows[win_num][0]
			(x1,y1) = CentroidWindows[win_num][1]

			# initialize empty list for each of the window (reset each loop)
			centroid_x, centroid_y = [], []
			error_x, error_y = [], []

			# loop through each REDUCED image array
			for i in range(len(ReducedImageArrays)):

				# extract pixels within current window
				windowed_image_array = ReducedImageArrays[i][y0:y1,x0:x1]
				#plt.figure()
				#plt.imshow(windowed_image_array)
				#plt.show()

				# calculate the center-of-mass centorid for each extracted window array (and error)
				# output centroids are in the windowed coordinate system 
				cx,cy,sigx,sigy = centroid(windowed_image_array) 

				# convert centroid location back to the full-frame coordinate system and append to list
				centroid_x.append(x0+cx)
				centroid_y.append(y0+cy)
				error_x.append(sigx)
				error_y.append(sigy)

			# sanity check: make sure that list is correct length before continuing
			if len(ImageDates) != len(centroid_x):
				print("Error: Number of calculated centroids does not match number of images?!?")
				exit(1)
			
			# add centroids (full-frame coordinates) and errors to dataframe
			df["centroid_x%d" % (win_num+1)] = centroid_x
			df["centroid_y%d" % (win_num+1)] = centroid_y
			df["error_x%d" % (win_num+1)] = error_x
			df["error_y%d" % (win_num+1)] = error_y

		# populated CentroidDataFrame class variable using timestamps as index (legacy. matches OG RidgeTrack output)
		CentroidDataFrame = df.set_index("timestamp")

		'''
		Save dataframe as csv file
		Must first run LocateCentroid() class function
		'''
		path = OUT
		# if given path is a directory, output csv is saved at 'path/centroids.csv'
		if os.path.exists(path) and os.path.isdir(path):
			path = path+"/centroids.csv"

		# if path exists, prompt user about overwritting it
		if  os.path.exists(path) and os.path.isfile(path):
			res = input("%s exists. Overwrite? [y]/n " % path)
			if res == 'n' or res == 'N':
				exit(0)

		# ensure file extension is .csv
		path = path.split(".")[0] + ".csv"
		
		CentroidDataFrame.to_csv(path, index_label='timestamp')
		print("centroid data written to %s" % path)


		'''
		Save dataframe as csv file
		Must first run LocateCentroid() class function
		'''
		path = OUT
		# if given path is a directory, output gif is saved at 'path/replay.gif'
		if os.path.exists(path) and os.path.isdir(path):
			path = path+"/replay.gif"

		# if path exists, prompt user about overwritting it
		if  os.path.exists(path) and os.path.isfile(path):
			res = input("gifs already exist. Overwrite? [y]/n ")
			if res == 'n' or res == 'N':
				exit(0)

		# ensure file extension is .gif
		path = path.split(".")[0] + ".gif"

		frames = []
		for image_array in ReducedImageArrays:
			frames.append(Image.fromarray(image_array.astype('uint8')).convert("RGB"))

		for i in range(len(frames)):
			draw_full = ImageDraw.Draw(frames[i])
			for win_num in range(len(CentroidWindows)):
				(x0,y0) = CentroidWindows[win_num][0]
				(x1,y1) = CentroidWindows[win_num][1]

				# PIL image coordinates are flipped from ndarray coordinates! (swap x and y)
				draw_full.rectangle((x0,y0,x1,y1), outline=(255,0,0))

				r=1
				cx = CentroidDataFrame.iloc[i]["centroid_x%d" % (win_num+1)]
				cy = CentroidDataFrame.iloc[i]["centroid_y%d" % (win_num+1)]
				draw_full.ellipse((cx-r,cy-r,cx+r,cy+r), fill=(255,0,0), outline=(255,0,0))

			draw_full.text((28, 36), ImageDates[i].strftime("%m/%d/%Y %H:%M:%S"), fill=(255, 0, 0))

		full_frame_one = frames[0]
		full_frame_one.save(path, format="GIF", append_images=frames,
				save_all=True, duration=100, loop=0)
		print("full dataset gif created at %s" % path)

		for win_num in range(centroid_num):
			window_frames = []
			(x0,y0) = CentroidWindows[win_num][0]
			(x1,y1) = CentroidWindows[win_num][1]
			w,h = x1-x0, y1-y0
			for frame in frames:
				window_frames.append(frame.crop((x0,y0,x1+1,y1+1)).resize((w*10,h*10)))
			window_path = "%s-w%d.gif" % (path[:-4], (win_num+1))
			window_frame_one = window_frames[0]
			window_frame_one.save(window_path, format="GIF", 
				append_images=window_frames, save_all=True, duration=100, loop=0)
			print("window %d gif created at %s" % ((win_num+1), window_path))
