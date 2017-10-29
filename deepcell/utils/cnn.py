#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

"""
cnn_functions.py

Functions for building and training convolutional neural networks
"""

import os
import datetime

import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.api.keras.layers import Layer, InputSpec
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow.contrib.keras.api.keras.activations as activations
import tensorflow.contrib.keras.api.keras.initializers as initializers
import tensorflow.contrib.keras.api.keras.regularizers as regularizers
import tensorflow.contrib.keras.api.keras.constraints as constraints
from tensorflow.contrib.keras.python.keras.utils import conv_utils

from .helper import *
from .data import *
from .image_generators import *

"""
Custom layers
"""

class dilated_MaxPool2D(Layer):
	def __init__(self, pool_size=(2, 2), strides=None, dilation_rate=1, padding='valid',
				data_format=None, **kwargs):
		super(dilated_MaxPool2D, self).__init__(**kwargs)
		data_format = conv_utils.normalize_data_format(data_format)
		if dilation_rate != 1:
			strides = (1, 1)
		elif strides is None:
			strides = (1, 1)
		self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
		self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
		self.dilation_rate = dilation_rate
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.input_spec = InputSpec(ndim=4)

	def compute_output_shape(self):
		if self.data_format == 'channels_first':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.data_format == 'channels_last':
			rows = input_shape[1]
			cols = input_shape[2]

		rows = conv_utils.conv_output_length(rows, pool_size[0], padding='valid', stride=self.strides[0], dilation=dilation_rate)
		cols = conv_utils.conv_output_length(cols, pool_size[1], padding='valid', stride=self.strides[1], dilation=dilation_rate)

		if self.data_format == 'channels_first':
			return (input_shape[0], input_shape[1], rows, cols)
		elif self.data_format == 'channels_last':
			return (input_shape[0], rows, cols, input_shape[3])

	def  _pooling_function(self, inputs, pool_size, dilation_rate, strides, padding, data_format):
		backend = K.backend()

		#dilated pooling for tensorflow backend
		if backend == "theano":
			Exception('This version of DeepCell only works with the tensorflow backend')

		if data_format == 'channels_first':
			df = 'NCHW'
		elif data_format == 'channel_last':
			df = 'NHWC'
		output = tf.nn.pool(inputs, window_shape=pool_size, pooling_type="MAX", padding="VALID",
							dilation_rate=(dilation_rate, dilation_rate), strides=strides, data_format=df)

		return output

	def call(self, inputs):
		output = self._pooling_function(inputs=inputs,
										pool_size=self.pool_size,
										strides=self.strides,
										dilation_rate=self.dilation_rate,
										padding=self.padding,
										data_format=self.data_format)
		return output

	def get_config(self):
		config = {'pool_size': self.pool_size,
					'padding': self.padding,
					'dilation_rate': self.dilation_rate,
					'strides': self.strides,
					'data_format': self.data_format}
		base_config = super(dilated_MaxPool2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class TensorProd2D(Layer):
	def __init__(self,
					input_dim,
					output_dim,
					data_format=None,
					activation=None,
					use_bias=True,
					kernel_initializer='glorot_uniform',
					bias_initializer='zeros',
					kernel_regularizer=None,
					bias_regularizer=None,
					activity_regularizer=None,
					kernel_constraint=None,
					bias_constraint=None,
					**kwargs):
		super(TensorProd2D, self).__init__(**kwargs)
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.activation = activations.get(activation)
		self.use_bias = use_bias
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.input_spec = InputSpec(min_ndim=2)

	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs should be defined. Found None')
		input_dim = input_shape[channel_axis]

		self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
										initializer=self.kernel_initializer,
										name='kernel',
										regularizer=self.kernel_regularizer,
										constraint=self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.output_dim,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None

		# Set input spec.
		self.input_spec = InputSpec(min_ndim=2,
									axes={channel_axis: input_dim})
		self.built = True

	def call(self, inputs):
		backend = K.backend()

		if backend == "theano":
			Exception('This version of DeepCell only works with the tensorflow backend')

		if self.data_format == 'channels_first':
			output = tf.tensordot(inputs, self.kernel, axes=[[1], [0]])
			output = tf.transpose(output, perm=[0, 3, 1, 2])
			# output = K.dot(inputs, self.kernel)

		elif self.data_format == 'channels_last':
			output = tf.tensordot(inputs, self.kernel, axes=[[3], [0]])

		if self.use_bias:
			output = K.bias_add(output, self.bias, data_format=self.data_format)

		if self.activation is not None:
			return self.activation(output)

		return output

	def compute_output_shape(self, input_shape):
		rows = input_shape[2]
		cols = output_shape[3]
		if self.data_format == 'channels_first':
			output_shape = tuple(input_shape[0], self.output_dim, rows, cols)

		elif self.data_format == 'channels_last':
			output_shape = tuple(input_shape[0], rows, cols, self.output_dim)

		return output_shape

	def get_config(self):
		config = {
			'input_dim': self.input_dim,
			'output_dim': self.output_dim,
			'data_format': self.data_format,
			'activation': self.activation,
			'use_bias': self.use_bias,
			'kernel_initializer': initializers.serialize(self.kernel_initializer),
			'bias_initializer': initializers.serialize(self.bias_initializer),
			'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer': regularizers.serialize(self.bias_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
			'bias_constraint': constraints.serialize(self.bias_constraint)
		}
		base_config = super(TensorProd2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

"""
Training convnets
"""

def train_model_sample(model=None, dataset=None, optimizer=None,
	expt="", it=0, batch_size=32, n_epoch=100,
	direc_save="/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/",
	direc_data="/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/",
	lr_sched=rate_scheduler(lr=0.01, decay=0.95),
	rotation_range=0, flip=True, shear=0, class_weight=None):

	training_data_file_name = os.path.join(direc_data, dataset + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

	train_dict, (X_test, Y_test) = get_data(training_data_file_name)

	# the data, shuffled and split between train and test sets
	print('X_train shape:', train_dict["channels"].shape)
	print(train_dict["pixels_x"].shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# determine the number of classes
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[1]

	print(output_shape, n_classes)

	# convert class vectors to binary class matrices
	train_dict["labels"] = to_categorical(train_dict["labels"], n_classes)
	Y_test = to_categorical(Y_test, n_classes)

	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	print('Using real-time data augmentation.')

	# this will do preprocessing and realtime data augmentation
	datagen = SampleDataGenerator(
		rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
		shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
		horizontal_flip=flip,  # randomly flip images
		vertical_flip=flip)  # randomly flip images

	# fit the model on the batches generated by datagen.flow()
	loss_history = model.fit_generator(datagen.sample_flow(train_dict, batch_size=batch_size),
						steps_per_epoch=len(train_dict["labels"])/batch_size,
						epochs=n_epoch,
						validation_data=(X_test, Y_test),
						validation_steps=X_test.shape[0]/batch_size,
						class_weight=class_weight,
						callbacks=[ModelCheckpoint(file_name_save, monitor='val_loss', verbose=0,
							save_best_only=True, mode='auto'), LearningRateScheduler(lr_sched)])

	np.savez(file_name_save_loss, loss_history=loss_history.history)

def train_model_fully_conv(model=None, dataset=None, optimizer=None,
	expt="", it=0, batch_size=1, n_epoch=100,
	direc_save="/home/vanvalen/ImageAnalysis/DeepCell2/trained_networks/",
	direc_data="/home/vanvalen/ImageAnalysis/DeepCell2/training_data_npz/",
	lr_sched=rate_scheduler(lr=0.01, decay=0.95),
	rotation_range=0, flip=True, shear=0, class_weight=None):

	training_data_file_name = os.path.join(direc_data, dataset + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it)  + ".h5")

	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_" + str(it) + ".npz")

	train_dict, (X_test, Y_test) = get_data(training_data_file_name, mode='conv')

	class_weight = class_weight #train_dict["class_weights"]
	# the data, shuffled and split between train and test sets
	print('Training data shape:', train_dict["channels"].shape)
	print('Training labels shape:', train_dict["labels"].shape)

	print('Testing data shape:', X_test.shape)
	print('Testing labels shape:', Y_test.shape)

	# determine the number of classes
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	print(output_shape, n_classes)

	class_weights = np.array([27.23, 6.12, 0.36], dtype=K.floatx())
	class_weights = np.array([1, 1, 1], dtype=K.floatx())
	def loss_function(y_true, y_pred):
		return categorical_crossentropy(y_true, y_pred, axis=3, class_weights=class_weights, from_logits=False)

	model.compile(loss=loss_function,
				  optimizer=optimizer,
				  metrics=['accuracy'])

	print('Using real-time data augmentation.')

	# this will do preprocessing and realtime data augmentation
	datagen = ImageFullyConvDataGenerator(
		rotation_range=rotation_range,  # randomly rotate images by 0 to rotation_range degrees
		shear_range=shear, # randomly shear images in the range (radians , -shear_range to shear_range)
		horizontal_flip=flip,  # randomly flip images
		vertical_flip=flip)  # randomly flip images

	datagen.flow(train_dict, batch_size=1).next()

	Y_test = np.rollaxis(Y_test, 1, 4)
	# y = np.rollaxis(y, 1, 4) #np.expand_dims(y, axis = 0)


	# fit the model on the batches generated by datagen.flow()

	# loss_history = model.fit(x = [x], y = [y], batch_size = 1, verbose = 1, epochs = 20, callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')])

	loss_history = model.fit_generator(datagen.flow(train_dict, batch_size=batch_size),
						steps_per_epoch=train_dict["labels"].shape[0]/batch_size,
						epochs=n_epoch,
						validation_data=(X_test, Y_test),
						validation_steps=X_test.shape[0]/batch_size,
						callbacks=[ModelCheckpoint(file_name_save, monitor='val_loss', verbose=1,
							save_best_only=True, mode='auto'), LearningRateScheduler(lr_sched)])

	model.save_weights(file_name_save)
	np.savez(file_name_save_loss, loss_history=loss_history.history)

	data_location = '/home/vanvalen/Data/RAW_40X_tube/set1/'
	channel_names = ["channel004", "channel001"]
	image_list = get_images_from_directory(data_location, channel_names)
	image = image_list[0]
	for j in xrange(image.shape[1]):
		image[0, j, :, :] = process_image(image[0, j, :, :], 30, 30, False)

	pred = model.predict(image)
	for j in xrange(3):
		save_name = 'feature_' +str(j) + '.tiff'
		tiff.imsave(save_name, pred[0, :, :, j])

	return model

"""
Running convnets
"""

def run_model(image, model, win_x=30, win_y=30, std=False, split=True, process=True):
	if process:
		for j in xrange(image.shape[1]):
			image[0, j, :, :] = process_image(image[0, j, :, :], win_x, win_y, std)

	if split:
		image_size_x = image.shape[2]/2
		image_size_y = image.shape[3]/2
	else:
		image_size_x = image.shape[2]
		image_size_y = image.shape[3]

	evaluate_model = K.function(
		[model.layers[0].input, K.learning_phase()],
		[model.layers[-1].output])

	n_features = model.layers[-1].output_shape[1]

	if split:
		model_output = np.zeros((n_features, 2*image_size_x-win_x*2, 2*image_size_y-win_y*2), dtype='float32')

		img_0 = image[:, :, 0:image_size_x+win_x, 0:image_size_y+win_y]
		img_1 = image[:, :, 0:image_size_x+win_x, image_size_y-win_y:]
		img_2 = image[:, :, image_size_x-win_x:, 0:image_size_y+win_y]
		img_3 = image[:, :, image_size_x-win_x:, image_size_y-win_y:]

		model_output[:, 0:image_size_x-win_x, 0:image_size_y-win_y] = evaluate_model([img_0, 0])[0]
		model_output[:, 0:image_size_x-win_x, image_size_y-win_y:] = evaluate_model([img_1, 0])[0]
		model_output[:, image_size_x-win_x:, 0:image_size_y-win_y] = evaluate_model([img_2, 0])[0]
		model_output[:, image_size_x-win_x:, image_size_y-win_y:] = evaluate_model([img_3, 0])[0]

	else:
		model_output = evaluate_model([image, 0])[0]
		model_output = model_output[0, :, :, :]

	model_output = np.pad(model_output, pad_width=((0, 0), (win_x, win_x), (win_y, win_y)), mode='constant', constant_values=0)
	return model_output

def run_model_on_directory(data_location, channel_names, output_location, model, win_x=30, win_y=30,
							std=False, split=True, process=True, save=True):

	n_features = model.layers[-1].output_shape[1]
	counter = 0

	image_list = get_images_from_directory(data_location, channel_names)
	processed_image_list = []

	for image in image_list:
		print("Processing image " + str(counter + 1) + " of " + str(len(image_list)))
		processed_image = run_model(image, model, win_x=win_x, win_y=win_y, std=std, split=split, process=process)
		processed_image_list += [processed_image]

		# Save images
		if save:
			for feat in xrange(n_features):
				feature = processed_image[feat, :, :]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_"+ str(counter) + r'.tif')
				tiff.imsave(cnnout_name, feature)
		counter += 1

	return processed_image_list

def run_models_on_directory(data_location, channel_names, output_location, model_fn, list_of_weights,
							n_features=3, image_size_x=1080, image_size_y=1280, win_x=30, win_y=30,
							std=False, split=True, process=True, save=True):

	if split:
		input_shape = (len(channel_names), image_size_x/2+win_x, image_size_y/2+win_y)
	else:
		input_shape = (len(channel_names), image_size_x, image_size_y)

	model = model_fn(input_shape=input_shape, n_features=n_features)

	n_features = model.layers[-1].output_shape[1]

	model_outputs = []
	for weights_path in list_of_weights:
		print(weights_path)
		model.load_weights(weights_path)
		processed_image_list = run_model_on_directory(data_location, channel_names, output_location,
							model, win_x=win_x, win_y=win_y, save=False, std=std,
							split=split, process=process)
		model_outputs += [np.stack(processed_image_list, axis=0)]

	# Average all images
	model_output = np.stack(model_outputs, axis=0)
	model_output = np.mean(model_output, axis=0)

	# Save images
	if save:
		for img in xrange(model_output.shape[0]):
			for feat in xrange(n_features):
				feature = model_output[img, feat, :, :]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_" + str(img) + r'.tif')
				tiff.imsave(cnnout_name, feature)

	return model_output

def make_parallel(model, gpu_count):
	def get_slice(data, idx, parts):
		shape = tf.shape(data)
		size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
		stride = tf.concat([shape[:1] // parts, shape[1:]*0], axis=0)
		start = stride * idx
		return tf.slice(data, start, size)

	outputs_all = []
	for i in range(len(model.outputs)):
		outputs_all.append([])

	#Place a copy of the model on each GPU, each getting a slice of the batch
	for i in range(gpu_count):
		with tf.device('/gpu:%d' % i):
			with tf.name_scope('tower_%d' % i) as scope:

				inputs = []
				#Slice each input into a piece for processing on this GPU
				for x in model.inputs:
					input_shape = tuple(x.get_shape().as_list())[1:]
					slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i, 'parts':gpu_count})(x)
					inputs.append(slice_n)

				outputs = model(inputs)

				if not isinstance(outputs, list):
					outputs = [outputs]

				#Save all the outputs for merging back together later
				for l in range(len(outputs)):
					outputs_all[l].append(outputs[l])

	# merge outputs on CPU
	with tf.device('/cpu:0'):
		merged = []
		for outputs in outputs_all:
			merged.append(merge(outputs, mode='concat', concat_axis=0))

	return Model(inputs=model.inputs, outputs=merged)
