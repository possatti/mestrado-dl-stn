#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import requests
import argparse
import shutil
import random
import json
import math
import sys
import re
import os

try:
    import pandas as pd
    USE_PANDAS = True
except:
    USE_PANDAS = False

try:
    import cPickle as pickle
except:
    import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'cifar10'))
import cifar_utils

print(sys.version, file=sys.stderr)
print('Numpy version:', np.version.version, file=sys.stderr)

np.random.seed(7)
random.seed(7)

CIFAR_10_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_10_TARGZ_FILEPATH = os.path.join(os.path.dirname(__file__), 'cifar-10-python.tar.gz')
CIFAR_10_UNPACKING_DIR = os.path.dirname(__file__)
CIFAR_10_DIR = os.path.join(os.path.dirname(__file__), 'cifar-10-batches-py')

CIFAR_10_DISTORTED_DIR = os.path.join(os.path.dirname(__file__), 'cifar-10-distorted')


def build(args):
    """Downloads CIFAR-10 and create distorted version."""

    # Download CIFAR 10 dataset.
    if not os.path.isfile(CIFAR_10_TARGZ_FILEPATH):
        print("Downloading CIFAR 10 to `{}`...".format(CIFAR_10_TARGZ_FILEPATH), file=sys.stderr)
        r = requests.get(CIFAR_10_DOWNLOAD_URL, stream=True)
        if r.ok:
            with open(CIFAR_10_TARGZ_FILEPATH, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    else:
        print("CIFAR 10 already downloaded to `{}`.".format(CIFAR_10_TARGZ_FILEPATH), file=sys.stderr)

    # Unpack the dataset.
    if not os.path.isdir(CIFAR_10_DIR):
        print("Unpacking CIFAR 10 to: `{}`...".format(CIFAR_10_DIR), file=sys.stderr)
        shutil.unpack_archive(CIFAR_10_TARGZ_FILEPATH, CIFAR_10_UNPACKING_DIR)
    else:
        print("CIFAR 10 already unpacked to: `{}`.".format(CIFAR_10_DIR), file=sys.stderr)

    # meta_path = os.path.join(CIFAR_10_DIR, 'batches.meta')
    # batch1_path = os.path.join(CIFAR_10_DIR, 'data_batch_1')

    # Mess with the test images.
    if not os.path.isdir(CIFAR_10_DISTORTED_DIR):
        print("Creating CIFAR-10-DISTORTED at: `{}`.".format(CIFAR_10_DISTORTED_DIR), file=sys.stderr)
        X, y = cifar_utils.load_batch(os.path.join(CIFAR_10_DIR, 'test_batch'))
        Xz, params = cifar_utils.distort_batch(X, horizontal_flip=False, return_parameters=True)
        cifar_utils.save_batch(os.path.join(CIFAR_10_DISTORTED_DIR, 'test_batch'), Xz, y)
        print("CIFAR-10-DISTORTED created.", file=sys.stderr)
        if USE_PANDAS:
            params_filepath = os.path.join(CIFAR_10_DISTORTED_DIR, 'test_batch_params.csv')
            print("Saving distortions parameters to `{}`...".format(params_filepath), file=sys.stderr)
            pd.DataFrame(params).to_csv(params_filepath, index=False)
    else:
        print("CIFAR-10-DISTORTED already present at: `{}`.".format(CIFAR_10_DISTORTED_DIR), file=sys.stderr)

def inspect(args):
    X, y = cifar_utils.load_batch(args.batch_path)
    bv = cifar_utils.BatchVisualizer(X, y, 2, 5)
    bv.show()

def check(args):
    import time
    import datetime
    test_batch_path = os.path.join(CIFAR_10_DIR, 'test_batch')
    print('Loading test batch from {}...'.format(test_batch_path), file=sys.stderr)
    X, y = cifar_utils.load_batch(test_batch_path)

    # 10000 images using pillow.
    print('Applying distortions...', file=sys.stderr)
    begin = time.time()
    Xz = cifar_utils.distort_batch(X, horizontal_flip=False)
    end = time.time()
    dt_using_pillow = datetime.timedelta(seconds=end-begin)
    print('It took {} to mess {} images, using pillow.'.format(dt_using_pillow, len(X)))

    # 128 images using pillow.
    begin = time.time()
    Xz = cifar_utils.distort_batch(X[:128,...], horizontal_flip=False)
    end = time.time()
    dt_using_pillow = datetime.timedelta(seconds=end-begin)
    print('It took {} to mess {} images, using pillow.'.format(dt_using_pillow, 128))

    # Compare original and distorted images, side by side.
    print('Showing results...', file=sys.stderr)
    for i in range(int(6*6/2)):
        plt.subplot(6, 6, i*2+1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.subplot(6, 6, i*2+2)
        plt.imshow(Xz[i])
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Arguments and options
    parser = argparse.ArgumentParser(description="""
        Downloads CIFAR-10 and create a version of the dataset,
        where the images are rotated and translated.""")

    subparsers = parser.add_subparsers(help='Available commands.', dest='command')
    build_parser = subparsers.add_parser('build', help='Build the CIFAR-10 distorted dataset.')
    check_parser = subparsers.add_parser('check', help='Check what the distortions are producing.')
    inspect_parser = subparsers.add_parser('inspect', help='Inspect batch.')
    inspect_parser.add_argument('batch_path')

    args = parser.parse_args()
    if args.command == 'build': build(args)
    elif args.command == 'inspect': inspect(args)
    elif args.command == 'check': check(args)
    else: parser.print_usage()
