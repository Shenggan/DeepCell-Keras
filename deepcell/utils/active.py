#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function

"""
cnn_functions.py

Functions for building and training convolutional neural networks
"""

from itertools import cycle
import numpy as np
import mahotas as mh

import matplotlib
matplotlib.use('TkAgg')
matplotlib.get_backend()

from skimage import morphology as morph
from skimage.segmentation import find_boundaries

from scipy.ndimage import binary_dilation, binary_erosion

"""
Active contours
"""

def zero_crossing(data, offset=0.5):
	new_data = data - offset
	zc = np.where(np.diff(np.signbit(new_data)))[0]
	return zc

"""
Define active contour functions
"""
class fcycle(object):

	def __init__(self, iterable):
		"""Call functions from the iterable each time it is called."""
		self.funcs = cycle(iterable)

	def __call__(self, *args, **kwargs):
		f = self.funcs.next()
		return f(*args, **kwargs)


# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3), np.array([[0, 1, 0]]*3), np.flipud(np.eye(3)), np.rot90([[0, 1, 0]]*3)]
_P3 = [np.zeros((3, 3, 3)) for i in xrange(9)]

_P3[0][:, :, 1] = 1
_P3[1][:, 1, :] = 1
_P3[2][1, :, :] = 1
_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_P3[8][[0, 1, 2], [2, 1, 0], :] = 1

_aux = np.zeros((0))
def SI(u):
	"""SI operator."""
	global _aux
	if np.ndim(u) == 2:
		P = _P2
	elif np.ndim(u) == 3:
		P = _P3
	else:
		raise ValueError, "u has an invalid number of dimensions (should be 2 or 3)"

	if u.shape != _aux.shape[1:]:
		_aux = np.zeros((len(P),) + u.shape)

	for i in xrange(len(P)):
		_aux[i] = binary_erosion(u, P[i])

	return _aux.max(0)

def IS(u):
	"""IS operator."""
	global _aux
	if np.ndim(u) == 2:
		P = _P2
	elif np.ndim(u) == 3:
		P = _P3
	else:
		raise ValueError, "u has an invalid number of dimensions (should be 2 or 3)"

	if u.shape != _aux.shape[1:]:
		_aux = np.zeros((len(P),) + u.shape)

	for i in xrange(len(P)):
		_aux[i] = binary_dilation(u, P[i])

	return _aux.min(0)

# SIoIS operator.
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = fcycle([SIoIS, ISoSI])

class MorphACWE(object):
	"""Morphological ACWE based on the Chan-Vese energy functional."""

	def __init__(self, data, smoothing=1, lambda1=10, lambda2=1):
		"""Create a Morphological ACWE solver.

		Parameters
		----------
		data : ndarray
			The image data.
		smoothing : scalar
			The number of repetitions of the smoothing step (the
			curv operator) in each iteration. In other terms,
			this is the strength of the smoothing. This is the
			parameter mu.
		lambda1, lambda2 : scalars
			Relative importance of the inside pixels (lambda1)
			against the outside pixels (lambda2).
		"""

		self._u = None
		self.smoothing = smoothing
		self.lambda1 = lambda1
		self.lambda2 = lambda2

		self.data = data
		self.mask = data

	def set_levelset(self, u):
		self._u = np.double(u)
		self._u[u > 0] = 1
		self._u[u <= 0] = 0

	levelset = property(lambda self: self._u,
						set_levelset,
						doc="The level set embedding function (u).")

	def step(self):
		"""Perform a single step of the morphological Chan-Vese evolution."""
		# Assign attributes to local variables for convenience.
		u = self._u

		if u is None:
			raise ValueError, "the levelset function is not set (use set_levelset)"

		data = self.data

		# Create mask to separate objects
		labeled, _ = mh.label(u)
		mask = mh.segmentation.gvoronoi(labeled)
		mask = 1-find_boundaries(mask)

		self.mask = np.float32(mask)/np.float32(mask).max()

		# Determine c0 and c1.
		inside = u > 0
		outside = u <= 0
		c0 = data[outside].sum() / float(outside.sum())
		c1 = data[inside].sum() / float(inside.sum())

		# Image attachment.
		dres = np.array(np.gradient(u))
		abs_dres = np.abs(dres).sum(0)
		aux = abs_dres * (self.lambda1*(data - c1)**2 - self.lambda2*(data - c0)**2)

		res = np.copy(u)
		res[aux < 0] = 1
		res[aux > 0] = 0

		# Smoothing.
		for _ in xrange(self.smoothing):
			res = curvop(res)

		# Apply mask
		res *= mask

		self._u = res

	def run(self, iterations):
		"""Run several iterations of the morphological Chan-Vese method."""
		for _ in xrange(iterations):
			self.step()

def segment_image_w_morphsnakes(img, nuc_label, num_iters, smoothing=2):

	morph_snake = MorphACWE(img, smoothing=smoothing, lambda1=1, lambda2=1)
	morph_snake.levelset = np.float16(nuc_label > 0)

	for _ in xrange(num_iters):
		morph_snake.step()

	seg_input = morph_snake.levelset
	seg = morph.watershed(seg_input, nuc_label, mask=(seg_input > 0))
	return seg
