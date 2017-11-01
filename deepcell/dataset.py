#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

"""
dataset.py

Executing functions for creating npz files containing the training data
Functions will create training dataset.

Files should be plased in training directories with each separate
dataset getting its own folder

"""

import os
import argparse

from utils.data import make_training_data_sample as make_training_data

def main():
	"""The Main Function."""
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataset", type=str, default="3T3", help="choose mode")
	parser.add_argument("-n", "--num_example", type=int,
						default=100000, help="the max number of training examples")
	parser.add_argument("-s", "--window_size", type=int, default=30, help="window size")
	args = parser.parse_args()

	# Define maximum number of training examples
	max_training_examples = args.num_example
	window_size = args.window_size

	# Load data
	root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
	direc_name = os.path.join(root, "DATA/train/" + args.dataset)
	file_name_save = os.path.join(root, 'DATA/train_npz/' + args.dataset + '.npz')
	training_direcs = os.listdir(direc_name)
	training_direcs = ["set1", "set2", "set3"]
	files = os.listdir(os.path.join(direc_name, training_direcs[0]))
	channel_names = [x[:-4] for x in files if 'feature' not in x]

	# Specify the number of feature masks that are present
	num_of_features = len([x for x in files if 'feature' in x])

	# Specify which feature is the edge feature
	edge_feature = [1, 0, 0]

	# Create the training data
	make_training_data(max_training_examples=max_training_examples,
						window_size_x=window_size,
						window_size_y=window_size,
						direc_name=direc_name,
						file_name_save=file_name_save,
						training_direcs=training_direcs,
						channel_names=channel_names,
						num_of_features=num_of_features,
						edge_feature=edge_feature,
						dilation_radius=1,
						sub_sample=True,
						display=False,
						verbose=True,
						process_std=True)

if __name__ == "__main__":
	main()
