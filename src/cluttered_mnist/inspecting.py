#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

print(sys.version, file=sys.stderr)
print('Numpy version:', np.version.version, file=sys.stderr)

def main(args):
    print('Loading Cluttered MNIST from: {}'.format(args.dataset), file=sys.stderr)
    mnist_cluttered = np.load(args.dataset)
    X = mnist_cluttered['X_test']
    y = mnist_cluttered['y_test']
    n_images = len(X)

    # Prepare images.
    X = X.reshape((len(X), 40, 40))
    X = X.astype('uint8')

    # Shuffle.
    np.random.seed(7)
    permutations = np.random.permutation(n_images)
    X = X[permutations]
    y = y[permutations]

    print("X.dtype:", X.dtype, file=sys.stderr) #!#
    print("X.shape:", X.shape, file=sys.stderr) #!#

    # Show images.
    for n in range(args.rows * args.columns):
        plt.subplot(args.rows, args.columns, n+1)
        plt.title(y[n][0])
        plt.imshow(X[n,...], cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    default_cluttered_mnist_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mnist_sequence1_sample_5distortions5x5.npz'))
    # Arguments and options.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default=default_cluttered_mnist_path)
    parser.add_argument('--rows', '-r', type=int, default=5)
    parser.add_argument('--columns', '-c', type=int, default=5)
    args = parser.parse_args()
    main(args)
