import shutil
import os, glob
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class DoolCentroid:

	def __init__(self, directory, centroid_num=2, input_dirname="IN",  reject_dirname="Reject"):
		
		self.directory = os.path.normpath(directory)
		self.centroid_num = centroid_num

		# first check if input paths exsist and are directories
		if not os.path.exists(self.directory) or not os.path.isdir(self.directory):
			print("%s does not exist or is not a directory" % self.directory)
			exit(1)

		# set input paths as class variables
		self.input_directory = os.path.join(self.directory, input_dirname) 
		self.reject_directory = os.path.join(self.directory, reject_dirname)

		if not os.path.exists(self.input_directory) or not os.path.isdir(self.input_directory):
			print("%s does not exist or is not a directory" % self.input_directory)
			exit(1)

		if not os.path.exists(self.reject_directory) or not os.path.isdir(self.reject_directory):
			res = input("%s does not exist or is not a directory. create? [y]/n " % self.reject_directory)
			if res == "n" or res == "N":
				exit(0)
			os.mkdir(self.reject_directory)

		# initialize class variables for sharing between class functions
		self.InitNumberImages = 0
		self.InitNumberRejected = 0
		self.ImagePaths = []         # list of image path names
		self.ImageDates = []         # list of image timestamps (from path names)
		self.ImageArrays = []        # list of 2D arrays for each images (default: red) 
		self.BackgroundArrays = []   # list of 2D background arrays (estimated)
		self.ReducedImageArrays = [] # list of 2D reduced images arrays (ImageArrays[i] - BackgroundArrays[i])

		self.CentroidWindows = []    # pixel window locations for centroiding ([x0, y0], [x1, y1])
		self.CentroidDataFrame = pd.DataFrame() # pandas dataframe for streamlined data viewing 

	###
	# Image Processing Class Functions:
	###

	def GetImages(self, image_color=0):
		# .jpgs are split into red/green/blue. Fire-i is monochrome so red=blue=green
		if image_color not in range(4):
			print("image color must be 0 (red), 1 (blue), 2 (green), or 3 (averaged)")
			exit(1)

		self.ClearImages() # reset class variables to empty

		# get list of filenames from 'IN' directory
		wildcard_jpgs_str = os.path.join(self.input_directory,"*.JPG")
		image_paths = glob.glob(wildcard_jpgs_str)
		if len(image_paths) == 0:
			print("No *.JPG files found in %s" % wildcard_jpgs_str)
			exit(0)

		df = pd.DataFrame()
		progbar = tqdm(image_paths, leave=False)
		for fpath in progbar:
			progbar.set_description("sorting frames")
			image_date  = self.getFileDate(fpath) # get date from filename 
			df = pd.concat([df, pd.DataFrame({'fpath': fpath}, index=[image_date])])
		df = df.sort_index()

		self.ImagePaths = df['fpath']
		self.ImageDates = df.index
		self.InitNumberImages = len(self.ImagePaths)

		start_str = df.index[0].strftime("%Y-%m-%d %H:%M:%S.%f")
		end_str   = df.index[-1].strftime("%Y-%m-%d %H:%M:%S.%f")
		hours = (df.index[-1] - df.index[0]).total_seconds()/3600

		print()
		print("%-12s %s" % ("directory:", self.input_directory))
		print("%-12s %s" % ("frames:", len(self.ImagePaths)))
		print("%-12s %s" % ("start:", start_str))
		print("%-12s %s" % ("end:", end_str))
		print("%-12s %.2f hours" % ("duration:", hours))

		print("\ncreating image arrays...")

		progbar = tqdm(range(len(self.ImagePaths)), leave=True)
		for i in progbar:
			image_date    = self.ImageDates[i]
			image_path    = self.ImagePaths[i]
			#image_name_sm = self.ImagePaths[i].split("/")[-1] # path w/out directory prefix 
			image_name_sm = os.path.basename(self.ImagePaths[i]) # path w/out directory prefix 
			progbar.set_description(image_name_sm)

			# get single color from jpg
			# Fire-i is monochrome so red=blue=green
			jpg_image = plt.imread(image_path,'F')

			image_array = jpg_image[:,:,image_color].astype('float64')

			if image_color == 3:
				for i in [1, 2]:
					image_array += jpg_image[:,:,i].astype('float64')
				image_array = image_array/3

			# append to image array and date to class variables
			self.ImageArrays.append(image_array)

	def EstimateBackground(self, filter_size_pixels=45, plot_prompt=True):
		'''
		Estimate background for images using large median filter.
		Populate BackgroundArrays and ReducedImageArrays class variables.
		Must first run GetImages() class function	
		'''

		# ensure that class variables are populated correctly beofre continuing
		if not self.ImagesReady():
			print("Images are not ready. Have you run GetImages() class function?")
			exit(1)

		print("\nestimating backgrounds and reducing images...")

		# initialize background and reduced image arrays with the same shape as the image arrays
		image_size = self.ImageArrays[0].shape
		bg_array = np.zeros(image_size)
		reduced_image_array = np.zeros(image_size)

		prev_mean = None
		prev_std = None

		progbar = tqdm(range(len(self.ImageArrays)), leave=True) 
		for i in progbar:
			new_background = False  # boolean date for estimating new background (reset each loop)
			show_plots = False      # boolean gate for plotting image reduction steps (reset each loop)

			image_array = self.ImageArrays[i]
			#image_name_sm = self.ImagePaths[i].split("/")[-1] # file name without directory prefix 
			image_name_sm = os.path.basename(self.ImagePaths[i]) # path w/out directory prefix 

			# simple stats for each image array
			mean = np.mean(image_array)
			std = np.std(image_array)

			if prev_mean == None:     # true only for first loop. 
				progbar.clear()
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
				print("(%d of %d): \033[1mestimating background\033[0m of %s (size=%d px)..." % (i+1, len(self.ImageArrays), image_name_sm, filter_size_pixels))

				# estimate background array with large median filter (>= twice centroid diameter of ~20px)
				bg_array = ndimage.median_filter(image_array, size=filter_size_pixels)

				if plot_prompt:
					# user input to view plots (plots called at end of current class function)
					res = input("Done! Plot Estimated Background? y/[n] ")
					if res == "y" or res == "Y":
						show_plots = True

			
			#print("(%d of %d): removing background from %s..." % (i+1, len(self.ImageArrays), image_name_sm,))
			progbar.set_description("%s" % (image_name_sm,))

			# subtract background from image array. small median filter (3px) to remove spurious pixels 
			reduced_image_array = ndimage.median_filter((image_array - bg_array), size=3)

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

	def CreateWindows(self, threshold=20, padding=5):
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
		reject_frame_indices = []

		print("\ncreating %d windows..." % (self.centroid_num))

		progbar = tqdm(range(len(self.ReducedImageArrays)), leave=True)
		for i in progbar:
			#image_name_sm = self.ImagePaths[i].split("/")[-1] # file name without directory prefix 
			image_name_sm = os.path.basename(self.ImagePaths[i]) # path w/out directory prefix 
			progbar.set_description("%s" % (image_name_sm))

			reduced_image_array = self.ReducedImageArrays[i]

			# all values below threshold (default=20 counts) set to zero
			threshold_array = reduced_image_array
			threshold_array[threshold_array < threshold] = 0

			# label and number the features
			lbl,num = ndimage.label(threshold_array)

			# ensure that the correct number of features are detected. 
			if num != self.centroid_num:
				msg = "%s rejected: %d features detected" % (image_name_sm, num)
				reject_frame_indices.append(i)
				print(msg)
				self.InitNumberRejected += 1
				continue

			# only if correct num of features:
			# find objects that remain after thresholding
			peaks = ndimage.find_objects(lbl)

			for p in peaks:
				dy, dx = p

				# determine window size for current feature 
				x0,y0 = dx.start, dy.start
				h,w = np.shape(self.ReducedImageArrays[i][p])
				x1,y1  = x0+w, y0+h
				
				# initial window size added to list
				# note that this
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
				min_x_diff, win_idx = 999, None
				for w in range(len(peak_windows)):
					prev_x0 = peak_windows[w][0][0]
					dif = abs(x0 - prev_x0)
					if dif < min_x_diff:
						min_x_diff, win_idx = dif, w
				
				# determine minimum x0,x0 and maximum x1,y1 so detected features accross all images are included in the each window area
				win_x0 = min(x0, peak_windows[win_idx][0][0])
				win_y0 = min(y0, peak_windows[win_idx][0][1])
				win_x1 = max(x1, peak_windows[win_idx][1][0])
				win_y1 = max(y1, peak_windows[win_idx][1][1])

				# set window values to be used during the next loop
				peak_windows[win_idx] = [(win_x0, win_y0),(win_x1, win_y1)]

		# sanity check (again)
		if len(peak_windows) != self.centroid_num:
			print("Error: Too many windows!?! (should not happen)")
			exit(1)

		for win_num in range(len(peak_windows)):
			(x0,y0) = peak_windows[win_num][0]
			(x1,y1) = peak_windows[win_num][1]
			peak_windows[win_num][0] = (x0-padding, y0-padding)
			peak_windows[win_num][1] = (x1+padding, y1+padding)

		# populate class variable for future class fucntions
		self.CentroidWindows = peak_windows

		# loop through images with incorecct centroids
		# must be done in reverse so indexes dont change each loop!
		for idx in sorted(reject_frame_indices, reverse=True):
			shutil.copy2(self.ImagePaths[idx], self.reject_directory)
		
		self.ImagePaths = [i for j, i in enumerate(self.ImagePaths) if j not in reject_frame_indices]
		self.ImageDates = [i for j, i in enumerate(self.ImageDates) if j not in reject_frame_indices]
		self.ImageArrays = [i for j, i in enumerate(self.ImageArrays) if j not in reject_frame_indices]
		self.BackgroundArrays = [i for j, i in enumerate(self.BackgroundArrays) if j not in reject_frame_indices]
		self.ReducedImageArrays = [i for j, i in enumerate(self.ReducedImageArrays) if j not in reject_frame_indices]

		# sanity check of array sizes!
		if len(self.ReducedImageArrays) != (self.InitNumberImages - self.InitNumberRejected):
			print("Error: reduced images array is unexpected size (should not happen!?!)")
		if not self.ImagesReady() or not self.ReducedImagesReady(): 
			print("Error: arrays are not compatible sizes (should not happen!?!)")
		

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


		print()

		# initialize empty dataframe
		df = pd.DataFrame()
		# add timestamps column to dataframe
		df["timestamp"] = self.ImageDates

		for win_num in range(len(self.CentroidWindows)):
			(x0,y0) = self.CentroidWindows[win_num][0]
			(x1,y1) = self.CentroidWindows[win_num][1]

			# initialize empty list for each of the window (reset each loop)
			centroid_x, centroid_y = [], []
			error_x, error_y = [], []

			print("calculating window %d centroids..." % (win_num+1))

			# loop through each REDUCED image array
			progbar = tqdm(range(len(self.ReducedImageArrays)), leave=True)
			for i in progbar:
				#image_name_sm = self.ImagePaths[i].split("/")[-1] # file name without directory prefix 
				image_name_sm = os.path.basename(self.ImagePaths[i]) # path w/out directory prefix 
				progbar.set_description("%s" % (image_name_sm))

				# extract pixels within current window
				windowed_image_array = self.ReducedImageArrays[i][y0:y1,x0:x1]

				# calculate the center-of-mass centorid for each extracted window array (and error)
				# output centroids are in the windowed coordinate system 
				cx,cy,sigx,sigy = self.centroid(windowed_image_array) 

				# convert centroid location back to the full-frame coordinate system and append to list
				centroid_x.append(x0+cx)
				centroid_y.append(y0+cy)
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

	###
	# output dataframe and create gifs
	###

	def CreateDataCSV(self, csv_path="centroids.csv"):
		'''
		Save dataframe as csv file
		Must first run LocateCentroid() class function
		'''

		# ensure that class variables are populated correctly before continuing
		if not self.CentroidDataReady():
			print("Centroid data not ready. Have you run LocateCentroids() class function?")
			return

		print("\ncreating centroid csv file...")

		out_path = os.path.join(self.directory, csv_path)
		file_split = os.path.splitext(out_path)
		if file_split[-1] != ".csv":
			out_path = "%s.csv" % file_split[0]


		# ensure file extension is .csv
		if out_path[-4:] != ".csv":
			print("out_path argument must be a directory or filename ending in '.csv'")
			exit(1)

		# if path exists, prompt user about overwritting it
		if os.path.exists(out_path) and os.path.isfile(out_path):
			res = input("%s exists. Overwrite? [y]/n " % out_path)
			if res == 'n' or res == 'N':
				return
		
		print("saving data to csv...")
		self.CentroidDataFrame.to_csv(out_path, index_label='timestamp')
		print("centroid data written to %s" % out_path)


	def CreateGifs(self, gif_path="replay.gif", duration_ms=50):
		'''
		create replay gifs for images and individual windows	
		Must run LocateCentroid() class function first
		'''

		# ensure that class variables are populated correctly before continuing
		if not self.CentroidDataReady():
			print("Centroid data not ready. Have you run LocateCentroids() class function?")
			return
		
		print("\ncreating centroid gif files...")

		out_path = os.path.join(self.directory, gif_path)
		file_split = os.path.splitext(out_path)
		if file_split[-1] != ".gif":
			out_path = "%s.gif" % file_split[0]

		# if path exists, prompt user about overwritting it
		if  os.path.exists(out_path) and os.path.isfile(out_path):
			res = input("gifs already exist. Overwrite? [y]/n ")
			if res == 'n' or res == 'N':
				return

		frames = []
		for image_array in self.ReducedImageArrays:
			frames.append(Image.fromarray(image_array.astype('uint8')).convert("RGB"))

		progbar = tqdm(range(len(frames)), leave=False)
		for i in progbar:
			progbar.set_description("annotating full-frame images for gif")
			draw_full = ImageDraw.Draw(frames[i])
			for win_num in range(len(self.CentroidWindows)):
				(x0,y0) = self.CentroidWindows[win_num][0]
				(x1,y1) = self.CentroidWindows[win_num][1]

				# PIL image coordinates are flipped from ndarray coordinates! (swap x and y)
				draw_full.rectangle((x0,y0,x1,y1), outline=(255,0,0))

				r=1
				cx = self.CentroidDataFrame.iloc[i]["centroid_x%d" % (win_num+1)]
				cy = self.CentroidDataFrame.iloc[i]["centroid_y%d" % (win_num+1)]
				draw_full.ellipse((cx-r,cy-r,cx+r,cy+r), fill=(255,0,0), outline=(255,0,0))

			draw_full.text((28, 36), self.ImageDates[i].strftime("%m-%d-%Y %H:%M:%S"), fill=(255, 0, 0))

		print("saving full-frame gif...")
		full_frame_one = frames[0]
		full_frame_one.save(out_path, format="GIF", append_images=frames,
				save_all=True, duration=duration_ms, loop=0)
		print("full dataset gif created at %s" % out_path)

		for win_num in range(self.centroid_num):
			window_frames = []
			(x0,y0) = self.CentroidWindows[win_num][0]
			(x1,y1) = self.CentroidWindows[win_num][1]
			w,h = x1-x0, y1-y0

			progbar = tqdm(range(len(frames)), leave=False)
			for i in progbar:
				progbar.set_description("annotating window %d images for gif")
				# crop and resize the pil images. integer can be fixed by not using resize()
				window_frames.append(frames[i].crop((x0,y0,x1+1,y1+1)).resize((w*10,h*10)))
				draw_window = ImageDraw.Draw(window_frames[i])
				draw_window.text((28, 36), self.ImageDates[i].strftime("%m-%d-%Y %H:%M:%S"), fill=(255, 0, 0))


			file_split = os.path.splitext(out_path)
			window_out_path = "%s-w%d.gif" % (file_split[0], (win_num+1))
			print("saving window %d gif..." % (win_num+1))
			window_frame_one = window_frames[0]
			window_frame_one.save(window_out_path, format="GIF", 
				append_images=window_frames, save_all=True, duration=duration_ms, loop=0)
			print("window %d gif created at %s" % ((win_num+1), window_out_path))


	####
	# Utility class functions:
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

	def getFileDate(self, filename):
		NAME = filename.split('_') # split filename string
		year = NAME[4]
		month = NAME[3]
		day  = NAME[2]
		hour = NAME[5] 
		minute = NAME[6] 
		second = NAME[7] 
		millisec = NAME[8].split(".")[0] 
		# create date string 
		datestr = '{:s}-{:s}-{:s} {:s}:{:s}:{:s}.{:s}'.format(year, month, day, hour, minute, second, millisec)
		# date string to datetime value
		return dt.datetime.strptime(datestr,'%Y-%m-%d %H:%M:%S.%f')

	def DisplayWindows(self):
		# ensure that class variables are populated correctly before continuing
		if not self.CentroidWindowsReady():
			print("Centroid windows are not ready. Have you run CreateWindows() class function?")
			exit(1)

		print()
		# display information about each of the windows
		for win_num in range(len(self.CentroidWindows)):
			x_start = self.CentroidWindows[win_num][0][0]
			y_start = self.CentroidWindows[win_num][0][1]
			x_end = self.CentroidWindows[win_num][1][0]
			y_end = self.CentroidWindows[win_num][1][1]
			x_size = x_end - x_start
			y_size = y_end - y_start
			print("window_%d: (x0,y0) = (%d, %d), (x1,y1) = (%d, %d), (w,h) = (%d, %d)" % ((win_num+1), x_start, y_start, x_end, y_end, x_size, y_size))

	def BackgroundReductionPlot(self, idx):
		fig, ax = plt.subplots(1,3, sharex=True, sharey=True)

		ax[0].imshow(self.ImageArrays[idx], vmax=255)
		ax[1].imshow(self.BackgroundArrays[idx], vmax=255)
		ax[2].imshow(self.ReducedImageArrays[idx], vmax=255)

		ax[0].set_title("Original")
		ax[1].set_title("Background")
		ax[2].set_title("Reduced Image")
		for a in ax:
			a.grid(True, alpha=0.25)
		plt.show()

	def SingleFramePlot(self, idx):

		# ensure that class variables are populated correctly before continuing
		if not self.CentroidDataReady():
			print("Centroid data not ready. Have you run LocateCentroids() class function?")
			return

		fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
		ax.imshow(self.ReducedImageArrays[idx], vmax=255)

		for win_num in range(self.centroid_num):
			ax.plot(self.CentroidDataFrame["centroid_x%d" % (win_num+1)][idx], self.CentroidDataFrame["centroid_y%d" % (win_num+1)][idx],'r.')

	def ClearImages(self):
		self.InitNumberImages = 0
		self.InitNumberRejected = 0
		self.ImagePaths = [] 
		self.ImageDates = [] 
		self.ImageArrays = []
		self.BackgroundArrays = []
		self.ReducedImageArrays = []
		self.CentroidWindows = []
		self.CentroidDataFrame = pd.DataFrame()

	###
	# Boolean gate class functions
	###

	def CentroidDataReady(self):
		if len(self.CentroidDataFrame) == 0:
			return False
		if len(self.ImagePaths) != len(self.CentroidDataFrame):
			return False
		return True

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
