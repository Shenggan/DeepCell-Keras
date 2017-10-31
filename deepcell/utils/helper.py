#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

"""
helper_functions.py

Functions for making training data

@author: David Van Valen
"""

"""
Import python packages
"""

import os
import re
import numpy as np

import tifffile.tifffile as tiff
from skimage.io import imread
from scipy import ndimage

import tensorflow as tf
from keras import backend as K
import keras.activations as activations

def cf(x, y, sample_image):

	numrows, numcols = sample_image.shape
	col = int(x+0.5)
	row = int(y+0.5)
	if col >= 0 and col < numcols and row >= 0 and row < numrows:
		z = sample_image[row, col]
		return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
	return 'x=%1.4f, y=1.4%f'%(x, y)

def axis_softmax(x, axis=1):
	return activations.softmax(x, axis=axis)

def rotate_array_0(arr):
	return arr

def rotate_array_90(arr):
	axes_order = range(arr.ndim - 2) + [arr.ndim-1, arr.ndim-2]
	slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None), slice(None, None, -1)]
	return arr[tuple(slices)].transpose(axes_order)

def rotate_array_180(arr):
	slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None, None, -1), slice(None, None, -1)]
	return arr[tuple(slices)]

def rotate_array_270(arr):
	axes_order = range(arr.ndim-2) + [arr.ndim-1, arr.ndim-2]
	slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None, None, -1), slice(None)]
	return arr[tuple(slices)].transpose(axes_order)

def to_categorical(y, num_classes=None):
	"""Converts a class vector (integers) to binary class matrix.
	E.g. for use with categorical_crossentropy.
	# Arguments
		y: class vector to be converted into a matrix
		(integers from 0 to num_classes).
		num_classes: total number of classes.
	# Returns
		A binary matrix representation of the input.
	"""
	y = np.array(y, dtype='int').ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes))
	categorical[np.arange(n), y] = 1
	return categorical


def normalize(x, axis=-1, order=2):
	"""Normalizes a Numpy array.
	# Arguments
		x: Numpy array to normalize.
		axis: axis along which to normalize.
		order: Normalization order (e.g. 2 for L2 norm).
	# Returns
		A normalized copy of the array.
	"""
	l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
	l2[l2 == 0] = 1
	return x / np.expand_dims(l2, axis)

def get_image_sizes(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]
	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	return img_temp.shape

def rate_scheduler(lr=.001, decay=0.95):
	def output_fn(epoch):
		epoch = np.int(epoch)
		new_lr = lr * (decay ** epoch)
		return new_lr
	return output_fn

def process_image(channel_img, win_x, win_y, std=False, remove_zeros=False):
	if std:
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		std = np.std(channel_img)
		channel_img /= std
		return channel_img

	if remove_zeros:
		channel_img /= 255
		avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
		channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
		return channel_img

	p50 = np.percentile(channel_img, 50)
	channel_img /= p50
	avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
	channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
	return channel_img

def get_image(file_name):
	if '.tif' in file_name:
		im = np.float32(tiff.TIFFfile(file_name).asarray())
	else:
		im = np.float32(imread(file_name))
	return im

def format_coord(x, y, sample_image):
	numrows, numcols = sample_image.shape
	col = int(x+0.5)
	row = int(y+0.5)
	if col >= 0 and col < numcols and row >= 0 and row < numrows:
		z = sample_image[row, col]
		return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
	return 'x=%1.4f, y=1.4%f'%(x, y)

def nikon_getfiles(direc_name, channel_name):
	imglist = os.listdir(direc_name)
	imgfiles = [i for i in imglist if channel_name in i]

	def sorted_nicely(l):
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		return sorted(l, key=alphanum_key)

	imgfiles = sorted_nicely(imgfiles)
	return imgfiles

def get_images_from_directory(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]

	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	n_channels = len(channel_names)
	all_images = []

	for stack_iteration in xrange(len(img_list_channels[0])):
		all_channels = np.zeros((1, n_channels, img_temp.shape[0], img_temp.shape[1]), dtype='float32')
		for j in xrange(n_channels):
			channel_img = get_image(os.path.join(data_location, img_list_channels[j][stack_iteration]))
			all_channels[0, j, :, :] = channel_img
		all_images += [all_channels]

	return all_images

def _to_tensor(x, dtype):
	"""Convert the input `x` to a tensor of type `dtype`.
	# Arguments
		x: An object to be converted (numpy array, list, tensors).
		dtype: The destination type.
	# Returns
		A tensor.
	"""
	x = tf.convert_to_tensor(x)
	if x.dtype != dtype:
		x = tf.cast(x, dtype)
	return x

def categorical_crossentropy(target, output, class_weights=None, axis=None, from_logits=False):
	"""Categorical crossentropy between an output tensor and a target tensor.
	# Arguments
		target: A tensor of the same shape as `output`.
		output: A tensor resulting from a softmax
		(unless `from_logits` is True, in which
		case `output` is expected to be the logits).
		from_logits: Boolean, whether `output` is the
		result of a softmax, or is a tensor of logits.
	# Returns
		Output tensor.
	"""
	# Note: tf.nn.softmax_cross_entropy_with_logits
	# expects logits, Keras expects probabilities.
	if axis is None:
		axis = len(output.get_shape()) - 1
	if not from_logits:
		# scale preds so that the class probas of each sample sum to 1
		output /= tf.reduce_sum(output, axis=axis, keep_dims=True)
		# manual computation of crossentropy
		_epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
		if class_weights is None:
			return -tf.reduce_sum(target * tf.log(output), axis=axis)
		return -tf.reduce_sum(tf.multiply(target * tf.log(output), class_weights), axis=axis)

	return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)
