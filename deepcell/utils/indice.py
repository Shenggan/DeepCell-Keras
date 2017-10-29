"""
Helper functions for jaccard and dice indices
"""
from skimage import morphology as morph
from skimage.measure import label, regionprops
import numpy as np

def dice_jaccard_indices(mask, val, nuc_mask):

	strel = morph.disk(1)
	val = morph.erosion(val, strel)
	mask = mask.astype('int16')
	val = val.astype('int16')

	mask_label = label(mask, background=0)
	val_label = label(val, background=0)

	for j in xrange(1, np.amax(val_label)+1):
		if np.sum((val_label == j) * nuc_mask) == 0:
			val_label[val_label == j] = 0

	val_label = label(val_label > 0, background=0)

	mask_region = regionprops(mask_label)
	val_region = regionprops(val_label)

	jac_list = []
	dice_list = []

	for val_prop in val_region:
		temp = val_prop['coords'].tolist()
		internal_points_1 = set([tuple(l) for l in temp])
		best_mask_prop = mask_region[0]
		best_overlap = 0
		best_sum = 0
		best_union = 0

		for mask_prop in mask_region:
			temp = mask_prop['coords'].tolist()
			internal_points_2 = set([tuple(l) for l in temp])

			overlap = internal_points_1 & internal_points_2
			num_overlap = len(overlap)

			if num_overlap > best_overlap:
				best_mask_prop = mask_prop
				best_overlap = num_overlap
				best_union = len(internal_points_1 | internal_points_2)
				best_sum = len(internal_points_1) + len(internal_points_2)

		jac = np.float32(best_overlap)/np.float32(best_union)
		dice = np.float32(best_overlap)*2/best_sum

		if np.isnan(jac) == 0 and np.isnan(dice) == 0:
			jac_list += [jac]
			dice_list += [dice]

	JI = np.mean(jac_list)
	DI = np.mean(dice_list)
	print jac_list, dice_list
	print "Jaccard index is " + str(JI) + " +/- " + str(np.std(jac_list))
	print "Dice index is " + str(DI)  + " +/- " + str(np.std(dice_list))

	return JI, DI
