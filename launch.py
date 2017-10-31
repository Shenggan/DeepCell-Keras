#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

"""
launch.py

A launch scripts to start train or test.

Run command:
	python launch.py pipeline -d 3T3

"""

import os
import argparse

def main():
	"""The Main Function."""

	parser = argparse.ArgumentParser(description="A launch scripts to start train or test.")
	parser.add_argument("model")
	parser.add_argument("-d", "--dataset", type=str, default="3T3", help="choose mode")
	parser.add_argument("-n", "--num_example", type=int,
						default=100000, help="the max number of training examples")
	parser.add_argument("-s", "--window_size", type=int, default=30, help="window size")

	parser.add_argument("-b", "--batch_size", type=int,
						default=256, help="batch size")
	parser.add_argument("-e", "--n_epoch", type=int, default=1, help="how many epochs in training")
	parser.add_argument("--network", type=str,
						default="bn_feature_net_61x61", help="the network you choose")
	parser.add_argument("--n_model", type=int,
						default=1, help="how many models you want to train")
	parser.add_argument("--n_channels", type=int,
						default=2, help="must be same with your dataset")
	parser.add_argument("--n_features", type=int,
						default=3, help="must be num_of_features in dataset.py plus 1")

	parser.add_argument("--cyto_prefix", type=str,
					default="2017-10-29_3T3_bn_feature_net_61x61_", help="the prefix of your cyto modle")
	parser.add_argument("--nuclear_prefix", type=str,
					default="2017-10-29_nuclei_bn_feature_net_61x61_", help="the prefix of your nuclear modle")
	parser.add_argument("--win_cyto", type=int,
					default=30, help="window size of cyto model")
	parser.add_argument("--win_nuclear", type=int,
					default=30, help="window size of nuclear model")

	args = parser.parse_args()

	command_d = "python deepcell/dataset.py"
	command_d += " -n " + str(args.num_example)
	command_d += " -s " + str(args.window_size)

	command_d_nuclear = command_d + " -d nuclei"
	command_d += " -d " + args.dataset

	command_train = "python deepcell/train.py"
	command_train += " -b " + str(args.batch_size)
	command_train += " -e " + str(args.n_epoch)
	command_train += " --network " + args.network
	command_train += " --n_model " + str(args.n_model)
	command_train += " --n_features " + str(args.n_features)

	command_train_cyto = command_train + " --n_channels " + str(args.n_channels)
	command_train_cyto += " -d " + args.dataset
	command_train_nuclear = command_train + " -d nuclei" + " --n_channels 1"

	command_test_val = "python deepcell/test.py"
	command_test_val += " --n_model " + str(args.n_model)
	command_test_val += " --cyto_prefix " + args.cyto_prefix
	command_test_val += " --nuclear_prefix " + args.nuclear_prefix
	command_test_val += " --win_cyto " + str(args.win_cyto)
	command_test_val += " --win_nuclear " + str(args.win_nuclear)

	command_val = command_test_val + " -v 1" + " -d " + args.dataset
	command_test = command_test_val + " -v 0" + " -d " + args.dataset+'/set1'

	if args.model == "data_prepare" or args.model == "pipeline":
		print("Cyto Data Prepareing...")
		os.system(command_d)
		print("Nuclear Data Prepareing...")
		os.system(command_d_nuclear)
	if args.model == "train" or args.model == "pipeline":
		print("Training Cyto Model...")
		os.system(command_train_cyto)
		print("Training Nuclear Model...")
		os.system(command_train_nuclear)
	if args.model == "validation" or args.model == "pipeline":
		print("Validation...")
		os.system(command_val)
	if args.model == "test" or args.model == "pipeline":
		print("Testing...")
		os.system(command_test)

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	main()
