#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
 - https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
"""

from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

from cifar_utils import load_batch, BatchVisualizer

print(sys.version, file=sys.stderr)
print('Numpy version:', np.version.version, file=sys.stderr)


def main(args):
    test_batch_path = os.path.join(args.cifar10_dir, 'test_batch')
    X_test, y_test = load_batch(test_batch_path)

    # bv = BatchVisualizer(X_test, y_test, 5, 5)
    # bv.show()

    from keras.backend import set_image_data_format
    set_image_data_format('channels_last')

    from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rotation_range=60)
    datagen.fit(X_test)
    for X, y in datagen.flow(X_test, y_test, batch_size=36):
        bv = BatchVisualizer(X.astype(np.uint8), y.astype(np.uint8), 6, 6)
        bv.show()
        break

if __name__ == '__main__':
    # Defaults.
    default_cifar10_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cifar-10-batches-py'))

    # Arguments and options.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cifar10-dir', default=default_cifar10_path)
    args = parser.parse_args()
    main(args)
