#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

"""
model_zoo.py

Assortment of CNN architectures for single cell segmentation

@author: David Van Valen
"""

import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPool2D, Activation, Lambda, Flatten, BatchNormalization, Permute, Input, Concatenate
from tensorflow.contrib.keras.api.keras.regularizers import l2
from .cnn import dilated_MaxPool2D, TensorProd2D, axis_softmax

"""
Batch normalized conv-nets
"""

def bn_feature_net_21x21(n_features=3, n_channels=1, reg=1e-5, init='he_normal'):
	print("Using feature net 21x21 with batch normalization")
	model = Sequential()
	model.add(Conv2D(32, (4, 4), kernel_initializer=init, padding='valid', input_shape=(n_channels, 21, 21), kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(32, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(32, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(Flatten())

	model.add(Activation('softmax'))

	return model

def dilated_bn_feature_net_21x21(input_shape=(2, 1080, 1280), n_features=3, n_channels=1, reg=1e-5, init='he_normal', weights_path=None, from_logits=False):
	print("Using dilated feature net 21x21 with batch normalization")
	model = Sequential()
	d = 1
	model.add(Conv2D(32, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', input_shape=(n_channels, 21, 21), kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))

	if from_logits is True:
		model.add(Permute((1, 3, 4, 2)))

	if from_logits is False:
		model.add(Flatten())
		model.add(Activation(axis_softmax))

	if weights_path is not None:
		model.load_weights(weights_path, by_name=True)

	return model

def bn_feature_net_31x31(n_features=3, n_channels=1, reg=1e-5, init='he_normal'):
	print("Using feature net 31x31 with batch normalization")
	model = Sequential()
	model.add(Conv2D(32, (4, 4), kernel_initializer=init, padding='valid', input_shape=(n_channels, 31, 31), kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(Flatten())

	model.add(Activation('softmax'))

	return model

def dilated_bn_feature_net_31x31(n_features=3, n_channels=1, reg=1e-5, init='he_normal', from_logits=False, weights_path=None):
	print("Using dilated feature net 31x31 with batch normalization")
	model = Sequential()
	d = 1
	model.add(Conv2D(32, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', input_shape=(n_channels, 31, 31), kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(128, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))

	if from_logits is True:
		model.add(Permute((1, 3, 4, 2)))

	if from_logits is False:
		model.add(Activation(axis_softmax))

	if weights_path is not None:
		model.load_weights(weights_path, by_name=True)

	return model

def bn_feature_net_61x61(n_features=3, n_channels=1, reg=1e-5, init='he_normal'):
	print("Using feature net 61x61 with batch normalization")

	model = Sequential()
	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', input_shape=(n_channels, 61, 61), kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(200, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(Flatten())

	model.add(Activation('softmax'))

	return model

def dilated_bn_feature_net_61x61(input_shape=(2, 1080, 1280), batch_size=None, n_features=3, reg=1e-5, init='he_normal', weights_path=None, permute=False):
	print("Using dilated feature net 61x61 with batch normalization")

	model = Sequential()
	d = 1
	model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', input_shape=input_shape, batch_size=batch_size, kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(200, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(Activation(axis_softmax))

	if permute:
		model.add(Permute((2, 3, 1)))

	if weights_path is not None:
		model.load_weights(weights_path, by_name=True)

	return model

def bn_feature_net_81x81(n_features=3, n_channels=1, reg=1e-5, init='he_normal'):
	print("Using feature net 81x81 with batch normalization")

	model = Sequential()
	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', input_shape=(n_channels, 81, 81), kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(MaxPool2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(Flatten())

	model.add(Activation('softmax'))

	return model

def dilated_bn_feature_net_81x81(input_shape=(2, 1080, 1280), n_features=3, n_channels=1, reg=1e-5, init='he_normal', weights_path=None, from_logits=False):
	print("Using dilated feature net 81x81 with batch normalization")

	model = Sequential()
	d = 1
	model.add(Conv2D(64, (3, 3), dilution_rate=d, kernel_initializer=init, padding='valid', input_shape=(n_channels, 81, 11), kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (4, 4), dilution_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilution_rate=d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilution_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), dilution_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), dilution_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(dilution_rate=d, pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilution_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(64, (3, 3), dilution_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))
	model.add(dilated_MaxPool2D(pool_size=(2, 2)))
	d *= 2

	model.add(Conv2D(64, (3, 3), dilution_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(Conv2D(200, (4, 4), dilution_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg)))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('relu'))

	model.add(TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg)))

	if from_logits is True:
		model.add(Permute((1, 3, 4, 2)))

	if from_logits is False:
		model.add(Activation(axis_softmax))

	if weights_path is not None:
		model.load_weights(weights_path, by_name=True)

	return model

"""
Multi-resolution batch normalized conv-nets
"""

def bn_multires_feature_net_61x61(n_features=3, n_channels=1, reg=1e-5, init='he_normal'):
	print("Using multi-resolution feature net 61x61 with batch normalization")

	inputs = Input(shape=(n_channels, 61, 61))
	conv1 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', input_shape=(n_channels, 61, 61), kernel_regularizer=l2(reg))(inputs)
	norm1 = BatchNormalization(axis=1)(conv1)
	act1 = Activation('relu')(norm1)

	conv2 = Conv2D(64, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act1)
	norm2 = BatchNormalization(axis=1)(conv2)
	act2 = Activation('relu')(norm2)
	pool1 = MaxPool2D(pool_size=(2, 2))(act2)

	conv3 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
	norm3 = BatchNormalization(axis=1)(conv3)
	act3 = Activation('relu')(norm3)

	conv4 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act3)
	norm4 = BatchNormalization(axis=1)(conv4)
	act4 = Activation('relu')(norm4)
	pool2 = MaxPool2D(pool_size=(2, 2))(act4)

	conv5 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
	norm5 = BatchNormalization(axis=1)(conv5)
	act5 = Activation('relu')(norm5)

	conv6 = Conv2D(64, (3, 3), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act5)
	norm6 = BatchNormalization(axis=1)(conv6)
	act6 = Activation('relu')(norm6)
	pool3 = MaxPool2D(pool_size=(2, 2))(act6)

	side_conv1 = Conv2D(64, (28, 28), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
	side_norm1 = BatchNormalization(axis=1)(side_conv1)
	side_act1 = Activation('relu')(side_norm1)

	side_conv2 = Conv2D(64, (12, 12), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
	side_norm2 = BatchNormalization(axis=1)(side_conv2)
	side_act2 = Activation('relu')(side_norm2)

	side_conv3 = Conv2D(64, (4, 4), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool3)
	side_norm3 = BatchNormalization(axis=1)(side_conv3)
	side_act3 = Activation('relu')(side_norm3)

	merge_layer1 = Concatenate(axis=1)([side_act1, side_act2, side_act3])

	tensor_prod1 = TensorProd2D(192, 256, kernel_initializer=init, kernel_regularizer=l2(reg))(merge_layer1)
	norm7 = BatchNormalization(axis=1)(tensor_prod1)
	act7 = Activation('relu')(norm7)

	tensor_prod2 = TensorProd2D(256, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act7)
	flat = Flatten()(tensor_prod2)
	act8 = Activation('softmax')(flat)

	model = Model(inputs=inputs, outputs=act8)

	return model

def dilated_bn_multires_feature_net_61x61(input_shape=(2, 1080, 1280), n_features=3, reg=1e-5, init='he_normal', permute=False, weights_path=None, from_logits=False):
	print("Using dilated multi-resolution feature net 61x61 with batch normalization")

	d = 1
	inputs = Input(shape=input_shape)
	conv1 = Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(inputs)
	norm1 = BatchNormalization(axis=1)(conv1)
	act1 = Activation('relu')(norm1)

	conv2 = Conv2D(32, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act1)
	norm2 = BatchNormalization(axis=1)(conv2)
	act2 = Activation('relu')(norm2)
	pool1 = dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2))(act2)
	d *= 2

	conv3 = Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
	norm3 = BatchNormalization(axis=1)(conv3)
	act3 = Activation('relu')(norm3)

	conv4 = Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act3)
	norm4 = BatchNormalization(axis=1)(conv4)
	act4 = Activation('relu')(norm4)
	pool2 = dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2))(act4)
	d *= 2

	conv5 = Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
	norm5 = BatchNormalization(axis=1)(conv5)
	act5 = Activation('relu')(norm5)

	conv6 = Conv2D(32, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act5)
	norm6 = BatchNormalization(axis=1)(conv6)
	act6 = Activation('relu')(norm6)
	pool3 = dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2))(act6)
	d *= 2

	side_conv0 = Conv2D(32, (59, 59), dilation_rate=1, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(conv1)
	side_norm0 = BatchNormalization(axis=1)(side_conv0)
	side_act0 = Activation('relu')(side_norm0)

	side_conv1 = Conv2D(32, (28, 28), dilation_rate=2, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
	side_norm1 = BatchNormalization(axis=1)(side_conv1)
	side_act1 = Activation('relu')(side_norm1)

	side_conv2 = Conv2D(32, (12, 12), dilation_rate=4, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
	side_norm2 = BatchNormalization(axis=1)(side_conv2)
	side_act2 = Activation('relu')(side_norm2)

	side_conv3 = Conv2D(32, (4, 4), dilation_rate=8, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool3)
	side_norm3 = BatchNormalization(axis=1)(side_conv3)
	side_act3 = Activation('relu')(side_norm3)

	merge_layer1 = Concatenate(axis=1)([side_act1, side_act2, side_act3])

	tensor_prod1 = TensorProd2D(128, 64, kernel_initializer=init, kernel_regularizer=l2(reg))(merge_layer1)
	norm7 = BatchNormalization(axis=1)(tensor_prod1)
	act7 = Activation('relu')(norm7)

	tensor_prod2 = TensorProd2D(64, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act7)
	act8 = Activation(axis_softmax)(tensor_prod2)

	if permute:
		final_layer = Permute((2, 3, 1))(act8)

	model = Model(inputs=inputs, outputs=final_layer)

	if weights_path is not None:
		model.load_weights(weights_path, by_name=True)

	return model

def bn_multires_feature_net(input_shape=(2, 1080, 1280), n_features=3, reg=1e-5, init='he_normal', permute=False):
	input1 = Input(shape=input_shape)
	conv1 = Conv2D(16, (3, 3), dilation_rate=1, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(input1)
	norm1 = BatchNormalization(axis=1)(conv1)
	act1 = Activation('relu')(norm1)

	conv2 = Conv2D(16, (3, 3), dilation_rate=1, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act1)
	norm2 = BatchNormalization(axis=1)(conv2)
	act2 = Activation('relu')(norm2)

	conv3 = Conv2D(16, (3, 3), dilation_rate=2, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act2)
	norm3 = BatchNormalization(axis=1)(conv3)
	act3 = Activation('relu')(norm3)

	conv4 = Conv2D(16, (3, 3), dilation_rate=2, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act3)
	norm4 = BatchNormalization(axis=1)(conv4)
	act4 = Activation('relu')(norm4)

	conv5 = Conv2D(16, (3, 3), dilation_rate=4, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act4)
	norm5 = BatchNormalization(axis=1)(conv5)
	act5 = Activation('relu')(norm5)

	conv6 = Conv2D(16, (3, 3), dilation_rate=4, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act5)
	norm6 = BatchNormalization(axis=1)(conv6)
	act6 = Activation('relu')(norm6)

	conv7 = Conv2D(16, (3, 3), dilation_rate=8, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act6)
	norm7 = BatchNormalization(axis=1)(conv7)
	act7 = Activation('relu')(norm7)

	conv8 = Conv2D(16, (3, 3), dilation_rate=8, kernel_initializer=init, padding='same', kernel_regularizer=l2(reg))(act7)
	norm8 = BatchNormalization(axis=1)(conv8)
	act8 = Activation('relu')(norm8)

	merge1 = Concatenate(axis=1)([act1, act2, act3, act4, act5, act6, act7, act8])

	tensor_prod1 = TensorProd2D(16*8, 128, kernel_initializer=init, kernel_regularizer=l2(reg))(merge1)
	norm9 = BatchNormalization(axis=1)(tensor_prod1)
	act9 = Activation('relu')(norm9)

	tensor_prod2 = TensorProd2D(128, 128, kernel_initializer=init, kernel_regularizer=l2(reg))(act9)
	norm10 = BatchNormalization(axis=1)(tensor_prod2)
	act10 = Activation('relu')(norm10)

	tensor_prod3 = TensorProd2D(128, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act10)
	act11 = Activation(axis_softmax)(tensor_prod3)

	if permute:
		final_layer = Permute((2, 3, 1))(act11)

	model = Model(inputs=input1, outputs=final_layer)

	return model

"""
Multiple input conv-nets for fully convolutional training
"""

def dilated_bn_feature_net_gather_61x61(input_shape=(2, 1080, 1280), training_examples=1e5, batch_size=None, n_features=3, reg=1e-5, init='he_normal', weights_path=None, permute=False):
	print("Using dilated feature net 61x61 with batch normalization")

	input1 = Input(shape=input_shape)

	d = 1
	conv1 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', batch_size=batch_size, kernel_regularizer=l2(reg))(input1)
	norm1 = BatchNormalization(axis=1)(conv1)
	act1 = Activation('relu')(norm1)

	conv2 = Conv2D(64, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act1)
	norm2 = BatchNormalization(axis=1)(conv2)
	act2 = Activation('relu')(norm2)
	pool1 = dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2))(act2)
	d *= 2

	conv3 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool1)
	norm3 = BatchNormalization(axis=1)(conv3)
	act3 = Activation('relu')(norm3)

	conv4 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act3)
	norm4 = BatchNormalization(axis=1)(conv4)
	act4 = Activation('relu')(norm4)
	pool2 = dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2))(act4)
	d *= 2

	conv5 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool2)
	norm5 = BatchNormalization(axis=1)(conv5)
	act5 = Activation('relu')(norm5)

	conv6 = Conv2D(64, (3, 3), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(act5)
	norm6 = BatchNormalization(axis=1)(conv6)
	act6 = Activation('relu')(norm6)
	pool3 = dilated_MaxPool2D(dilation_rate=d, pool_size=(2, 2))(act6)
	d *= 2

	conv7 = Conv2D(200, (4, 4), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(pool3)
	norm7 = BatchNormalization(axis=1)(conv7)
	act7 = Activation('relu')(norm7)

	tensorprod1 = TensorProd2D(200, 200, kernel_initializer=init, kernel_regularizer=l2(reg))(act7)
	norm8 = BatchNormalization(axis=1)(tensorprod1)
	act8 = Activation('relu')(norm8)

	tensorprod2 = TensorProd2D(200, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(act8)
	act9 = Activation(axis_softmax)(tensorprod2)

	permute1 = Permute((2, 3, 1))(act9)

	batch_index_input = Input(batch_shape=(training_examples,), dtype='int32')
	row_index_input = Input(batch_shape=(training_examples,), dtype='int32')
	col_index_input = Input(batch_shape=(training_examples,), dtype='int32')

	index1 = K.stack([batch_index_input, row_index_input, col_index_input], axis=1)

	def gather_indices(x):
		return tf.gather_nd(x, index1)

	gather1 = Lambda(gather_indices)(permute1)

	model = Model(inputs=[input1, batch_index_input, row_index_input, col_index_input], outputs=[gather1])

	print(model.output_shape)

	return model
