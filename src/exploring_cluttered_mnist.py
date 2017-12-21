#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import sys
import re
import os

print(sys.version, file=sys.stderr)
print('Numpy version:', np.version.version, file=sys.stderr)

def main(args):
	print('Loading Cluttered MNIST from: {}'.format(args.dataset), file=sys.stderr)
	mnist_cluttered = np.load(args.dataset)

	X_train = mnist_cluttered['X_train']
	y_train = mnist_cluttered['y_train']
	X_valid = mnist_cluttered['X_valid']
	y_valid = mnist_cluttered['y_valid']
	X_test = mnist_cluttered['X_test']
	y_test = mnist_cluttered['y_test']

	np.random.seed(7)
	np.random.shuffle(X_test)

	subplots_xnumber = 5
	subplots_number = subplots_xnumber ** 2
	for page in range(int(1000 / subplots_number)):
		print("page:", page + 1, file=sys.stderr)
		fig, axes = plt.subplots(subplots_xnumber, subplots_xnumber)
		for i in range(subplots_xnumber):
			for j in range(subplots_xnumber):
				n = page*subplots_number + i*subplots_xnumber+j
				axes[i,j].imshow(mnist_cluttered['X_test'].reshape(1000, 40, 40)[n,...], cmap='gray')
				axes[i,j].axis('off')
		plt.show()



if __name__ == '__main__':
	# Defaults.
	default_cluttered_mnist_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist_sequence1_sample_5distortions5x5.npz'))

	# Arguments and options.
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--dataset', default=default_cluttered_mnist_path)
	args = parser.parse_args()
	main(args)