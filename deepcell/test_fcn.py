#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

"""
test.py
Run a trained CNN on a dataset.

Run command:
	python test.py

"""

import os
import argparse

import numpy as np
import scipy

import matplotlib.pyplot as plt

from skimage import segmentation
from skimage.measure import label

from utils.helper import nikon_getfiles, get_image, get_image_sizes
from utils.cnn import run_models_on_directory
from utils.segmentation import segment_nuclei, segment_cytoplasm
from utils.indice import dice_jaccard_indices

from utils.model import dilated_bn_multires_feature_net_61x61 as fcn_network
from utils.model import dilated_bn_feature_net_61x61 as network

from keras import backend as K
K.manual_variable_initialization(False)

def main():
	"""The Main Function."""
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--validation", type=int, default=1, help="turn on to cal the index")
	parser.add_argument("-d", "--dataset", type=str, default="3T3", help="choose mode")
	parser.add_argument("--n_model", type=int,
						default=1, help="how many models you want to train")
	parser.add_argument("--cyto_prefix", type=str,
					default="2017-10-29_3T3_bn_feature_net_61x61_", help="the prefix of your cyto modle")
	parser.add_argument("--nuclear_prefix", type=str,
					default="2017-10-29_nuclei_bn_feature_net_61x61_", help="the prefix of your nuclear modle")
	parser.add_argument("--win_cyto", type=int,
					default=30, help="window size of cyto model")
	parser.add_argument("--win_nuclear", type=int,
					default=30, help="window size of nuclear model")
	args = parser.parse_args()

	root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	if args.validation:
		direc_name = os.path.join(root, "DATA/validation/" + args.dataset)
	else:
		direc_name = os.path.join(root, "DATA/test/" + args.dataset)

	data_location = os.path.join(direc_name, 'RawImages')
	cyto_location = os.path.join(direc_name, 'Cytoplasm')
	nuclear_location = os.path.join(direc_name, 'Nuclear')
	mask_location = os.path.join(direc_name, 'Masks')

	cyto_channel_names = [x[:-4] for x in os.listdir(data_location)]
	nuclear_channel_names = [x[:-4] for x in os.listdir(data_location) if 'hase' not in x]

	trained_network_cyto_dir = os.path.join(root, "MODEL/cyto")
	trained_network_nuclear_dir = os.path.join(root, "MODEL/nuclear")

	n_model = args.n_model

	cyto_prefix = args.cyto_prefix
	nuclear_prefix = args.nuclear_prefix

	win_cyto = args.win_cyto
	win_nuclear = args.win_nuclear

	image_size_x, image_size_y = get_image_sizes(data_location, cyto_channel_names)

	"""
	Define model
	"""

	list_of_cyto_weights = []
	for j in xrange(n_model):
		cyto_weights = os.path.join(trained_network_cyto_dir, cyto_prefix + str(j) + ".h5")
		list_of_cyto_weights += [cyto_weights]

	list_of_nuclear_weights = []
	for j in xrange(n_model):
		nuclear_weights = os.path.join(trained_network_nuclear_dir, nuclear_prefix + str(j) + ".h5")
		list_of_nuclear_weights += [nuclear_weights]


	"""
	Run model on directory
	"""

	cytoplasm_predictions = run_models_on_directory(data_location, cyto_channel_names, cyto_location,
			n_features=3, model_fn=fcn_network, list_of_weights=list_of_cyto_weights, image_size_x=image_size_x,
			image_size_y=image_size_y, win_x=win_cyto, win_y=win_cyto, std=True, split=True)

	run_models_on_directory(data_location, nuclear_channel_names, nuclear_location,
			model_fn=network, list_of_weights=list_of_nuclear_weights,
			image_size_x=image_size_x, image_size_y=image_size_y,
			win_x=win_nuclear, win_y=win_nuclear, std=False, split=False)

	"""
	Refine segmentation with active contours
	"""

	nuclear_masks = segment_nuclei(img=None, color_image=True, load_from_direc=nuclear_location,
			mask_location=mask_location, area_threshold=100, solidity_threshold=0, eccentricity_threshold=1)

	cytoplasm_masks = segment_cytoplasm(img=None, load_from_direc=cyto_location, color_image=True,
			nuclear_masks=nuclear_masks, mask_location=mask_location, smoothing=1, num_iters=120)

	"""
	Compute validation metrics (optional)
	"""
	if args.validation:
		direc_val = os.path.join(direc_name, 'Validation')
		imglist_val = nikon_getfiles(direc_val, 'interior')

		val_name = os.path.join(direc_val, imglist_val[0])
		print(val_name)
		val = get_image(val_name)
		val = val[win_cyto:-win_cyto, win_cyto:-win_cyto]
		cyto = cytoplasm_masks[0, win_cyto:-win_cyto, win_cyto:-win_cyto]
		nuc = nuclear_masks[0, win_cyto:-win_cyto, win_cyto:-win_cyto]
		print(val.shape, cyto.shape, nuc.shape)

		print(dice_jaccard_indices(cyto, val, nuc))

	# Compute cell categorization prediction for each cell

	interior1 = cytoplasm_predictions[0, 1, :, :]
	interior2 = cytoplasm_predictions[0, 2, :, :]
	seg = label(cytoplasm_masks[0, :, :])
	num_of_cells = np.amax(seg)
	prediction = np.zeros(interior1.shape, dtype=np.float32)
	prediction_color = np.zeros((interior1.shape[0], interior1.shape[1], 3), dtype=np.float32)

	bound = segmentation.find_boundaries(seg)
	for cell_no in xrange(1, num_of_cells):
		class_1_pred = interior1[seg == cell_no]
		class_2_pred = interior2[seg == cell_no]
		# class_1_score = np.sum(class_1_pred) / (np.sum(class_1_pred) + np.sum(class_2_pred))
		class_2_score = np.sum(class_2_pred) / (np.sum(class_1_pred) + np.sum(class_2_pred))

		prediction[seg == cell_no] = class_2_score
		prediction_color[seg == cell_no, 0] = plt.cm.coolwarm(class_2_score)[0]
		prediction_color[seg == cell_no, 1] = plt.cm.coolwarm(class_2_score)[1]
		prediction_color[seg == cell_no, 2] = plt.cm.coolwarm(class_2_score)[2]

	prediction_color[bound, 0] = 0
	prediction_color[bound, 1] = 0
	prediction_color[bound, 2] = 0
	cnnout_name = os.path.join(mask_location, 'segmentation_rgb_new.tif')
	scipy.misc.imsave(cnnout_name, np.float16(prediction_color))

if __name__ == "__main__":
	main()
