#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

"""
train.py

Train a simple deep CNN on a dataset.

Run command:
	python train.py

"""

import os
import argparse

import tensorflow as tf
import keras
from keras import backend as K

from utils.cnn import rate_scheduler, train_model_sample as train_model
from utils.model import bn_multires_feature_net

def main():
	"""The Main Function."""
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataset", type=str, default="3T3", help="choose dataset")
	parser.add_argument("-b", "--batch_size", type=int,
						default=64, help="batch size")
	parser.add_argument("-e", "--n_epoch", type=int, default=1, help="how many epochs in training")
	parser.add_argument("-n", "--network", type=str,
						default="bn_feature_net_61x61", help="the network you choose")
	parser.add_argument("-m", "--n_model", type=int,
						default=1, help="how many models you want to train")
	parser.add_argument("-c", "--n_channels", type=int,
						default=2, help="must be same with your dataset")
	parser.add_argument("-f", "--n_features", type=int,
						default=3, help="must be num_of_features in dataset.py plus 1")
	parser.add_argument("--dist", type=int, default=0, help="1 to use distrbution training")
	args = parser.parse_args()

	if args.dist:
		print("Using Horovod to Distributed Training!")
		import horovod.tensorflow as hvd
		hvd.init()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.gpu_options.visible_device_list = str(hvd.local_rank())
		K.set_session(tf.Session(config=config))

	batch_size = args.batch_size
	n_epoch = args.n_epoch

	dataset = args.dataset
	expt = args.network

	root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	direc_save = os.path.join(root, "MODEL/cyto")
	if 'nuclei' in  args.dataset:
		direc_save = os.path.join(root, "MODEL/nuclear")
	direc_data = os.path.join(root, "DATA/train_npz")

	optimizer = tf.train.AdamOptimizer(learning_rate=0.5, beta1=0.9, beta2=0.999, epsilon=0.1, use_locking=False, name='Adam')
	if args.dist:
		optimizer = hvd.DistributedOptimizer(optimizer)

	optimizer = tf.train.AdamOptimizer(learning_rate=0.5, beta1=0.9, beta2=0.999, epsilon=0.1, use_locking=False, name='Adam')

	class_weights = {0:1e3, 1:1, 2:1}

	for iterate in xrange(args.n_model):
		model = bn_multires_feature_net(n_features=args.n_features, reg=1e-3, permute=True)
		train_model(model=model, dataset=dataset, optimizer=keras.optimizers.TFOptimizer(optimizer),
				expt=expt, it=iterate, batch_size=batch_size, n_epoch=n_epoch,
				direc_save=direc_save, direc_data=direc_data,
				class_weight=class_weights,
				rotation_range=0, flip=True, shear=False, dist=args.dist)

if __name__ == "__main__":
	main()
