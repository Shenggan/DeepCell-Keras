from helper import *
from active import *

import numpy as np
from numpy import array
import matplotlib
matplotlib.use('TkAgg')
matplotlib.get_backend()
import matplotlib.pyplot as plt
import shelve
from contextlib import closing

import os
import glob
import re
import numpy as np
import fnmatch
import tifffile.tifffile as tiff
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from scipy import ndimage
import threading
import scipy.ndimage as ndi
from scipy import linalg
import re
import random
import itertools
import h5py
import datetime
import scipy

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology as morph
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from skimage.filters import threshold_otsu
import skimage as sk
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils import class_weight




def segment_nuclei(img = None, save = True, adaptive = False, color_image = False, load_from_direc = None, feature_to_load = "feature_1", mask_location = None, threshold = 0.5, area_threshold = 50, eccentricity_threshold = 1, solidity_threshold = 0):
	# Requires a 4 channel image (number of frames, number of features, image width, image height)
	from skimage.filters import threshold_otsu, threshold_adaptive

	if load_from_direc is None:
		img = img[:,1,:,:]
		nuclear_masks = np.zeros(img.shape, dtype = np.float32)

	if load_from_direc is not None:
		img_files = nikon_getfiles(load_from_direc, feature_to_load )
		img_size = get_image_sizes(load_from_direc, feature_to_load)
		img = np.zeros((len(img_files), img_size[0], img_size[1]), dtype = np.float32)
		nuclear_masks = np.zeros((len(img_files), img.shape[1], img.shape[2]), dtype = np.float32)
		counter = 0
		for name in img_files:
			img[counter,:,:] = get_image(os.path.join(load_from_direc,name))
			counter += 1

	for frame in xrange(img.shape[0]):
		interior = img[frame,:,:]
		if adaptive:
			block_size = 61
			nuclear_mask = np.float32(threshold_adaptive(interior, block_size, method = 'median', offset = -.075))
		else: 
			nuclear_mask = np.float32(interior > threshold)
		nuc_label = label(nuclear_mask)
		max_cell_id = np.amax(nuc_label)
		for cell_id in xrange(1,max_cell_id + 1):
			img_new = nuc_label == cell_id
			img_fill = binary_fill_holes(img_new)
			nuc_label[img_fill == 1] = cell_id

		region_temp = regionprops(nuc_label)

		for region in region_temp:
			if region.area < area_threshold:
				nuclear_mask[nuc_label == region.label] = 0
			if region.eccentricity > eccentricity_threshold:
				nuclear_mask[nuc_label == region.label] = 0
			if region.solidity < solidity_threshold:
				nuclear_mask[nuc_label == region.label] = 0

		nuclear_masks[frame,:,:] = nuclear_mask

		if save:
			img_name = os.path.join(mask_location, "nuclear_mask_" + str(frame) + ".png")
			tiff.imsave(img_name,nuclear_mask)

		if color_image:
			img_name = os.path.join(mask_location, "nuclear_colorimg_" + str(frame) + ".png")
			
			from skimage.segmentation import find_boundaries
			import palettable
			from skimage.color import label2rgb

			seg = label(nuclear_mask)
			bound = find_boundaries(seg, background = 0)

			image_label_overlay = label2rgb(seg, bg_label = 0, bg_color = (0.8,0.8,0.8), colors = palettable.colorbrewer.sequential.YlGn_9.mpl_colors)
			image_label_overlay[bound == 1,:] = 0

			scipy.misc.imsave(img_name,np.float32(image_label_overlay))
	return nuclear_masks

def segment_cytoplasm(img =None, save = True, load_from_direc = None, feature_to_load = "feature_1", color_image = False, nuclear_masks = None, mask_location = None, smoothing = 1, num_iters = 80):
	if load_from_direc is None:
		cytoplasm_masks = np.zeros((img.shape[0], img.shape[2], img.shape[3]), dtype = np.float32)
		img = img[:,1,:,:]

	if load_from_direc is not None:
		img_files = nikon_getfiles(load_from_direc, feature_to_load )
		img_size = get_image_sizes(load_from_direc, feature_to_load)
		img = np.zeros((len(img_files), img_size[0], img_size[1]), dtype = np.float32)
		cytoplasm_masks = np.zeros((len(img_files), img.shape[1], img.shape[2]), dtype = np.float32)

		counter = 0
		for name in img_files:
			img[counter,:,:] = get_image(os.path.join(load_from_direc,name))
			counter += 1

	for frame in xrange(img.shape[0]):
		interior = img[frame,:,:]

		nuclei = nuclear_masks[frame,:,:]

		nuclei_label = label(nuclei, background = 0)

		seg = segment_image_w_morphsnakes(interior, nuclei_label, num_iters = num_iters, smoothing = smoothing)
		seg[seg == 0] = -1

		cytoplasm_mask = np.zeros(seg.shape,dtype = np.float32)
		max_cell_id = np.amax(seg)
		for cell_id in xrange(1,max_cell_id + 1):
			img_new = seg == cell_id
			img_fill = binary_fill_holes(img_new)
			cytoplasm_mask[img_fill == 1] = 1

		cytoplasm_masks[frame,:,:] = cytoplasm_mask

		if save:
			img_name = os.path.join(mask_location, "cytoplasm_mask_" + str(frame) + ".png")
			tiff.imsave(img_name,np.float32(cytoplasm_mask))

		if color_image:
			img_name = os.path.join(mask_location, "cytoplasm_colorimg_" + str(frame) + ".png")
			
			from skimage.segmentation import find_boundaries
			import palettable
			from skimage.color import label2rgb

			seg = label(cytoplasm_mask)
			bound = find_boundaries(seg, background = 0)

			image_label_overlay = label2rgb(seg, bg_label = 0, bg_color = (0.8,0.8,0.8), colors = palettable.colorbrewer.sequential.YlGn_9.mpl_colors)
			image_label_overlay[bound == 1,:] = 0

			scipy.misc.imsave(img_name,np.float32(image_label_overlay))

	return cytoplasm_masks