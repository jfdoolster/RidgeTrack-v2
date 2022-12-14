import shutil
import os, glob
import numpy as np
import pandas as pd
import datetime as dt
from scipy import ndimage
import matplotlib.pyplot as plt

class Centroid:

	def __init__(self, input_directory, reject_directory, expected_centroid_num=2):
		
		# first check if input paths exsist and are directories
		if not os.path.exists(input_directory) or not os.path.isdir(input_directory):
			print("`input_directory` path does not exist or is not a directory (%s)" % input_directory)
			exit(1)
		if not os.path.exists(reject_directory) or not os.path.isdir(reject_directory):
			print("`reject_directory` path does not exist or is not a directory" % reject_directory)
			exit(1)

		# set input paths as class variables
		self.IN = input_directory
		self.Reject = reject_directory 
		self.centroid_num = expected_centroid_num

		# initialize class variables for sharing between class functions
		self.ImagePaths = []         # list of image path names
		self.ImageDates = []         # list of image timestamps (from path names)
		self.ImageArrays = []        # list of 2D arrays for each images (default: red) 
		self.BackgroundArrays = []   # list of 2D background arrays (estimated)
		self.ReducedImageArrays = [] # list of 2D reduced images arrays (ImageArrays[i] - BackgroundArrays[i])

		self.CentroidWindows = []    # pixel window locations for centroiding ([x0, y0], [x1, y1])
		self.CentroidDataFrame = pd.DataFrame() # pandas dataframe for streamlined data viewing 

	def GetImages(self, image_color=0):
		'''
		Populate ImagePaths, ImageDates, and ImageArrays class variables. 
		Must run this class function first as other class functions require these class variables 
		'''
		# .jpgs are split into red/green/blue. Fire-i is monochrome so red=blue=green
		if image_color not in [0, 1, 2]:
			print("image color must be 0 (red), 1 (blue), or 2 (green)")
			exit(1)

		self.ClearImages() # reset class variables to empty

		# get sorted list of filenames from 'IN' directory
		self.ImagePaths = sorted(glob.glob(self.IN + "*.JPG"),key=os.path.getmtime)
		print("%d frames found in %s" % (len(self.ImagePaths), self.IN))

		for fname in self.ImagePaths:
			image_date  = self.getFileDate(fname) # get date from filename 

			# get single color from jpg
			# Fire-i is monochrome so red=blue=green
			jpg_image = plt.imread(fname,'F')
			image_array = jpg_image[:,:,image_color].astype('float64')

			# append to image array and date to class variables
			self.ImageArrays.append(image_array)
			self.ImageDates.append(image_date)	

	def EstimateBackground(self, filter_size_pixels=45):
		'''
		Estimate background for images using large median filter.
		Populate BackgroundArrays and ReducedImageArrays class variables.
		Must first run GetImages() class function	
		'''

		# ensure that class variables are populated correctly beofre continuing
		if not self.ImagesReady():
			print("Images are not ready. Have you run GetImages() class function?")
			exit(1)

		# initialize background and reduced image arrays with the same shape as the image arrays
		image_size = self.ImageArrays[0].shape
		bg_array = np.zeros(image_size)
		reduced_image_array = np.zeros(image_size)

		prev_mean = None
		prev_std = None

		for i in range(len(self.ImageArrays)):
			new_background = False  # boolean date for estimating new background (reset each loop)
			show_plots = False      # boolean gate for plotting image reduction steps (reset each loop)

			image_array = self.ImageArrays[i]
			image_name_sm = self.ImagePaths[i].split("/")[-1] # file name without directory prefix 

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

			# only estimate new background when needed (see above) to save processing time
			if new_background:
				print("Estimating background of %s (size=%d px)..." % (image_name_sm, filter_size_pixels))

				# estimate background array with large median filter (>= twice centroid diameter of ~20px)
				bg_array = ndimage.filters.median_filter(image_array, size=filter_size_pixels)

				# user input to view plots (plots called at end of current class function)
				res = input("Done! Plot Estimated Background? y/[n] ")
				if res == "y" or res == "Y":
					show_plots = True

			
			print("(%d of %d): Removing Background from %s..." % (i+1, len(self.ImageArrays), image_name_sm,))

			# subtract background from image array. small median filter (3px) to remove spurious pixels 
			reduced_image_array = ndimage.filters.median_filter((image_array - bg_array), size=3)

			# populate class variables
			self.BackgroundArrays.append(bg_array)
			self.ReducedImageArrays.append(reduced_image_array)

			# set simple stats for comparison in next loop
			prev_mean = mean
			prev_std = std

			if show_plots:
				# plot image, estimated background, and reduced image for verification
				self.BackgroundReductionPlot(i)  

				# user input to continue after viewing plots
				res = input("Continue? [y]/n ")
				if res == "n" or res == "N":
					exit(0)

	def CreateWindows(self, threshold=20):
		'''
		Create subwindows around centroids for unbiased centroiding. 	
		Image is thresholded and 'features' are found using scipy.ndimage functions
		Populates CentroidWindow class function.
		Must first run GetImages() and EstimateBackground() class functions
		'''

		# ensure that class variables are populated correctly before continuing
		if not self.ReducedImagesReady():
			print("Reduced Images are not ready. Have you run EstimateBackground() class function?")
			exit(1)

		# ensure window lists are empty before continuing
		self.CentroidWindows = [] 
		peak_windows = []

		for i in range(len(self.ReducedImageArrays)):
			reduced_image_array = self.ReducedImageArrays[i]

			# all values below threshold (default=20 counts) set to zero
			threshold_array = reduced_image_array
			threshold_array[threshold_array < threshold] = 0

			# label and number the features
			lbl,num = ndimage.label(threshold_array)

			# ensure that the correct number of features are detected. 
			if num != self.centroid_num:
				# remove image from future processing, copy original file to 'Reject' direcotry, and alert user
				self.RemoveSingleImage(i, "incorrect number of detected peaks (%d)" % num)
				continue

			# only if correct num of features:
			# find objects that remain after thresholding
			peaks = ndimage.find_objects(lbl)

			for p in peaks:
				dx, dy = p

				# determine window size for current feature 
				x0,y0 = dx.start, dy.start
				h,w = np.shape(self.ReducedImageArrays[i][p])
				x1,y1  = x0+w, y0+h
				
				# initial window size added to list
				if i == 0:
					peak_windows.append([(x0,y0),(x1,y1)])
					continue
				
				# sanity check, previous conditional should have removed images with incorrect number of features
				if len(peak_windows) != self.centroid_num:
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
		if len(peak_windows) != self.centroid_num:
			print("Error: Too many windows!?! (should not happen)")
			exit(1)

		padding = 10
		for win_num in range(len(peak_windows)):
			(x0,y0) = peak_windows[win_num][0]
			(x1,y1) = peak_windows[win_num][1]
			peak_windows[win_num][0] = (x0-padding, y0-padding)
			peak_windows[win_num][1] = (x1+padding, y1+padding)

		# display information about each of the windows
		for win_num in range(len(peak_windows)):
			x_start = peak_windows[win_num][0][0]
			y_start = peak_windows[win_num][0][1]
			x_size = peak_windows[win_num][1][0] - x_start
			y_size = peak_windows[win_num][1][1] - y_start
			print("window_%d: (x,y) = (%d, %d), (w,h) = (%d, %d)" % ((win_num+1), x_start, y_start, x_size, y_size))

		# populate class variable for future class fucntions
		self.CentroidWindows = peak_windows

	def LocateCentroids(self):
		'''
		Locate center-of-mass centroids using windows calculated in CreateWindows class function.
		Populate pandas dataframe ('CentroidDataFrame') class variable.
		Must first run CreateWindows() class function
		'''

		# ensure that class variables are populated correctly before continuing
		if not self.CentroidWindowsReady():
			print("Centroid windows are not ready. Have you run CreateWindows() class function?")
			exit(1)

		# initialize empty dataframe
		df = pd.DataFrame()

		# add timestamps column to dataframe
		df["timestamp"] = self.ImageDates

		for win_num in range(len(self.CentroidWindows)):
			x0 = self.CentroidWindows[win_num][0][0]
			y0 = self.CentroidWindows[win_num][0][1]
			x1 = self.CentroidWindows[win_num][1][0]
			y1 = self.CentroidWindows[win_num][1][1]

			# initialize empty list for each of the window (reset each loop)
			centroid_x, centroid_y = [], []
			error_x, error_y = [], []

			# loop through each REDUCED image array
			for i in range(len(self.ReducedImageArrays)):

				# extract pixels within current window
				windowed_image_array = self.ReducedImageArrays[i][x0:x1,y0:y1]

				# calculate the center-of-mass centorid for each extracted window array (and error)
				# output centroids are in the windowed coordinate system 
				cx,cy,sigx,sigy = self.centroid(windowed_image_array) 

				# convert centroid location back to the full-frame coordinate system and append to list
				centroid_x.append(cx+x0)
				centroid_y.append(cy+y0)
				error_x.append(sigx)
				error_y.append(sigy)

			# sanity check: make sure that list is correct length before continuing
			if len(self.ImageDates) != len(centroid_x):
				print("Error: Number of calculated centroids does not match number of images?!?")
				exit(1)
			
			# add centroids (full-frame coordinates) and errors to dataframe
			df["centroid_x%d" % (win_num+1)] = centroid_x
			df["centroid_y%d" % (win_num+1)] = centroid_y
			df["error_x%d" % (win_num+1)] = error_x
			df["error_y%d" % (win_num+1)] = error_y

		# populated CentroidDataFrame class variable using timestamps as index (legacy. matches OG RidgeTrack output)
		self.CentroidDataFrame = df.set_index("timestamp")

		# display dataframe in terminal
		print(self.CentroidDataFrame)

	def CreateDataCSV(self, csv_path):
		'''
		Save dataframe as csv file
		Must first run LocateCentroid() class function
		'''

		# if given path is a directory, output csv is saved at 'csv_path/centroids.csv'
		if os.path.exists(csv_path) and os.path.isdir(csv_path):
			csv_path = csv_path+"/centroids.csv"

		# if csv_path exists, prompt user about overwritting it
		if  os.path.exists(csv_path) and os.path.isfile(csv_path):
			res = input("%s exists. Overwrite? [y]/n" % csv_path)
			if res == 'n' or res == 'N':
				exit(0)
		
		self.CentroidDataFrame.to_csv(csv_path, index_label='timestamp')
		print("centroid data written to %s" % csv_path)


	def CreateGifs(self, gif_path):

		# if given path is a directory, output gif is saved at 'gif_path/replay.gif'
		if os.path.exists(gif_path) and os.path.isdir(gif_path):
			gif_path = gif_path+"/replay.gif"

		# if gif_path exists, prompt user about overwritting it
		if  os.path.exists(gif_path) and os.path.isfile(gif_path):
			res = input("%s exists. Overwrite? [y]/n" % gif_path)
			if res == 'n' or res == 'N':
				exit(0)

		from PIL import Image, ImageDraw

		frames = [Image.open(image) for image in self.ImagePaths]
		for i in range(len(frames)):
			draw = ImageDraw.Draw(frames[i])
			for win_num in range(len(self.CentroidWindows)):
				x0 = self.CentroidWindows[win_num][0][0]
				y0 = self.CentroidWindows[win_num][0][1]
				x1 = self.CentroidWindows[win_num][1][0]
				y1 = self.CentroidWindows[win_num][1][1]

				# PIL image coordinates are flipped from ndarray coordinates! (swap x and y)
				draw.rectangle((y0,x0,y1,x1))

			draw.text((28, 36), self.ImageDates[i].strftime("%m/%d/%Y %H:%M:%S"), fill=(255, 0, 0))


		frame_one = frames[0]
		frame_one.save(gif_path, format="GIF", append_images=frames,
				save_all=True, duration=100, loop=0)


	####
	# Utility functions:
	####

	def centroid(self,I): # calculate centroid and error (<x> and sigma_<x>)
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

	def BackgroundReductionPlot(self, idx):
		fig, ax = plt.subplots(1,3, sharex=True, sharey=True)

		ax[0].imshow(self.ImageArrays[idx], vmax=255)
		ax[1].imshow(self.BackgroundArrays[idx], vmax=255)
		ax[2].imshow(self.ReducedImageArrays[idx], vmax=255)

		ax[0].set_title("Original")
		ax[1].set_title("Background")
		ax[2].set_title("Reduced Image")
		plt.show()

	def CentroidWindowsReady(self):
		if len(self.CentroidWindows) != self.centroid_num:
			return False
		return True

	def ReducedImagesReady(self):
		if len(self.BackgroundArrays) == 0:
			return False
		if len(self.ImagePaths) != len(self.BackgroundArrays):
			return False
		if len(self.ImagePaths) != len(self.ReducedImageArrays):
			return False
		return True

	def ImagesReady(self):
		if len(self.ImagePaths) == 0: 
			return False
		if len(self.ImagePaths) != len(self.ImageDates): 
			return False
		if len(self.ImagePaths) != len(self.ImageArrays): 
			return False
		return True

	def RejectSingleImage(self, idx, msg=None):
		image_name_sm = self.ImagePaths[idx].split("/")[-1]
		shutil.copy2(self.ImagePaths[idx], self.Reject)
		self.ImagePaths.pop(idx)
		self.ImageDates.pop(idx) 
		self.ImageArrays.pop(idx)
		self.BackgroundArrays.pop(idx)
		self.ReducedImageArrays.pop(idx)
		if type(msg) == str:
			print("%s removed from analysis: %s" % (image_name_sm, msg))
		else:
			print("%s removed from analysis" % image_name_sm)

	def ClearImages(self):
		self.ImagePaths = [] 
		self.ImageDates = [] 
		self.ImageArrays = []
		self.BackgroundArrays = []
		self.ReducedImageArrays = []
		self.CentroidWindows = []
		self.CentroidDataFrame = pd.DataFrame()

	def getFileDate(self, filename):
		NAME = filename.split('_') # split filename string
		# create date string 
		datestr = '{:s}-{:s}-{:s} {:s}:{:s}:{:s}'.format(NAME[3],NAME[2],NAME[4],NAME[5],NAME[6],NAME[7])
		# date string to datetime value
		return dt.datetime.strptime(datestr,'%m-%d-%Y %H:%M:%S')



	

