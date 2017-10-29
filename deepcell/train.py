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

from tensorflow.contrib.keras.api.keras.optimizers import SGD

from utils.cnn import rate_scheduler, make_parallel, train_model_sample as train_model
from utils.model import bn_feature_net_61x61

def main():
	"""The Main Function."""
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataset", type=str, default="3T3", help="choose dataset")
	parser.add_argument("-b", "--batch_size", type=int,
						default=256, help="batch size")
	parser.add_argument("-e", "--n_epoch", type=int, default=1, help="how many epochs in training")
	parser.add_argument("-n", "--network", type=str,
						default="bn_feature_net_61x61", help="the network you choose")
	parser.add_argument("-m", "--n_model", type=int,
						default=1, help="how many models you want to train")
	parser.add_argument("-c", "--n_channels", type=int,
						default=2, help="must be same with your dataset")
	parser.add_argument("-f", "--n_features", type=int,
						default=3, help="must be num_of_features in dataset.py plus 1")
	args = parser.parse_args()

	batch_size = args.batch_size
	n_epoch = args.n_epoch

	dataset = args.dataset
	expt = args.network

	root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	direc_save = os.path.join(root, "MODEL/cyto")
	if 'nuclei' in  args.dataset:
		direc_save = os.path.join(root, "MODEL/nuclear")
	direc_data = os.path.join(root, "DATA/train_npz")

	optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	lr_sched = rate_scheduler(lr=0.01, decay=0.95)

	class_weights = {0:1, 1:1, 2:1}

	for iterate in xrange(args.n_model):
		model = bn_feature_net_61x61(n_channels=args.n_channels, n_features=args.n_features, reg=1e-5)
		make_parallel(model, 1)
		train_model(model=model, dataset=dataset, optimizer=optimizer,
				expt=expt, it=iterate, batch_size=batch_size, n_epoch=n_epoch,
				direc_save=direc_save, direc_data=direc_data,
				lr_sched=lr_sched, class_weight=class_weights,
				rotation_range=180, flip=True, shear=False)

if __name__ == "__main__":
	main()
