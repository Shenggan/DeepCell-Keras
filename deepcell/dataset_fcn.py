"""
dataset_fcn.py

Executing functions for creating npz files containing the training data
Functions will create training data for Fully convolutional training of single image conv-nets

Files should be placed in training directories with each separate
dataset getting its own folder

"""

import os

from utils.cnn import make_training_data_fully_conv as make_training_data

import matplotlib
matplotlib.use('TkAgg')
matplotlib.get_backend()

def main():
	"""The Main Function."""

	# Define maximum number of training examples
	max_training_examples = 1e6
	window_size = 30

	# Load data
	direc_name = '/home/csg/building/DeepCell/training_data/HeLa_joint/'
	file_name_save = os.path.join('/home/csg/building/DeepCell/training_data_npz/HeLa/', 'HeLa_conv.npz')
	training_direcs = ["set1", "set2", "set3", "set4", "set5"]
	channel_names = ["phase", "nuclear"]

	# Specify the number of feature masks that are present
	num_of_features = 2

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
						num_of_features=2,
						edge_feature=edge_feature,
						dilation_radius=2,
						sub_sample=False,
						display=True,
						verbose=True)

if __name__ == "__main__":
	main()
