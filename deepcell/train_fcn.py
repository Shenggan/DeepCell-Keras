"""
train_fcn.py

Train a simple deep CNN on a dataset in a fully convolutional fashion.

Run command:
	python train_fcn.py

"""

from __future__ import print_function

import os
import numpy as np

from tensorflow.contrib.keras.api.keras.optimizers import SGD, RMSprop

from utils.cnn import rate_scheduler, train_model_fully_conv as train_model
from utils.model import bn_multires_feature_net as the_model

def main():
	"""The Main Function."""

	batch_size = 1
	n_epoch = 40

	dataset = "HeLa_conv"
	expt = "bn_feature_net_61x61"

	direc_save = "/home/csg/building/DeepCell/trained_networks/HeLa/"
	direc_data = "/home/csg/building/DeepCell/training_data_npz/HeLa/"

	optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	lr_sched = rate_scheduler(lr=0.01, decay=0.99)

	file_name = os.path.join(direc_data, dataset + ".npz")
	training_data = np.load(file_name)
	class_weights = training_data["class_weights"]
	print(class_weights)
	class_weights = {0:1e3, 1:1, 2: 1}

	for iterate in xrange(1):

		model = the_model(input_shape=(2, 1080, 1280), n_features=3, reg=1e-3, permute=True)

		trained_model = train_model(model=model, dataset=dataset, optimizer=optimizer,
				expt=expt, it=iterate, batch_size=batch_size, n_epoch=n_epoch,
				direc_save=direc_save, direc_data=direc_data,
				lr_sched=lr_sched, class_weight=class_weights,
				rotation_range=0, flip=True, shear=False)

if __name__ == "__main__":
	main()
