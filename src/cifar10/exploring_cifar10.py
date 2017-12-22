#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

try:
    import cPickle as pickle
except:
    import pickle

print(sys.version, file=sys.stderr)
print('Numpy version:', np.version.version, file=sys.stderr)

def main(args):
    print('Loading CIFAR 10 on: `{}`.'.format(args.cifar10_dir))

    # Batch paths.
    meta_path = os.path.join(args.cifar10_dir, 'batches.meta')
    batch1_path = os.path.join(args.cifar10_dir, 'data_batch_1')
    batch2_path = os.path.join(args.cifar10_dir, 'data_batch_2')
    batch3_path = os.path.join(args.cifar10_dir, 'data_batch_3')
    batch4_path = os.path.join(args.cifar10_dir, 'data_batch_4')
    batch5_path = os.path.join(args.cifar10_dir, 'data_batch_5')
    test_batch_path = os.path.join(args.cifar10_dir, 'test_batch')

    # Open metadata.
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        label_names = meta[b'label_names']

    with open(batch1_path, 'rb') as f:
        batch1 = pickle.load(f, encoding='bytes')

    # X_batch1 = batch1[b'data'].reshape(10000, 32,32,3)
    X_batch1 = batch1[b'data'].reshape((10000, 32,32,3), order='F')
    X_batch1 = np.rot90(X_batch1, k=3, axes=(1,2))
    y_batch1 = batch1[b'labels']

    fig, axes = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            n = i*5+j
            axes[i,j].imshow(X_batch1[n,...])
            axes[i,j].axis('off')

    plt.imshow(X_batch1[0,...])
    plt.show()


if __name__ == '__main__':
    # Arguments and options.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cifar10-dir', default=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cifar-10-batches-py')))
    args = parser.parse_args()
    main(args)
