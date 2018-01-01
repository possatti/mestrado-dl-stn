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
    import cPickle as pickle
except:
    import pickle

print(sys.version, file=sys.stderr)
print('Numpy version:', np.version.version, file=sys.stderr)

np.random.seed(7)
random.seed(7)

CIFAR_10_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_10_TARGZ_FILEPATH = os.path.join(os.path.dirname(__file__), 'cifar-10-python.tar.gz')
CIFAR_10_UNPACKING_DIR = os.path.dirname(__file__)
CIFAR_10_DIR = os.path.join(os.path.dirname(__file__), 'cifar-10-batches-py')

CIFAR_10_ZUADO_DIR = os.path.join(os.path.dirname(__file__), 'cifar-10-zuado')

def zuar_batch(batch_path, dest_zuado):
    X, y = load_batch(batch_path)
    n_images = len(X)
    Xz = np.empty(shape=(n_images, 64,64,3), dtype=X.dtype)

    for i in range(n_images):
        # Define parameters for transformation.
        angle_deg = random.randint(-30,30)
        tx, ty = random.randint(-10,10), random.randint(-10,10)
        sx, sy = 1, 1

        # Transform image.
        im_m = X[i,...]
        im = Image.fromarray(im_m)
        newim = Image.new(im.mode, (64,64))
        newim.paste(im, box=(16,16))
        newim = newim.rotate(angle_deg)
        newim = newim.transform((64,64), Image.AFFINE, (sx,0,tx, 0,sy,ty))
        newim_m = np.asarray(newim, dtype=np.uint8).reshape(64,64,3)
        Xz[i] = newim_m

    # # Compare original and distorted images, side by side.
    # for i in range(int(6*6/2)):
    #     plt.subplot(6, 6, i*2+1)
    #     plt.imshow(X[i])
    #     plt.axis('off')
    #     plt.subplot(6, 6, i*2+2)
    #     plt.imshow(Xz[i])
    #     plt.axis('off')
    # plt.show()

    save_batch(dest_zuado, Xz, y)

def load_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data']
    y = batch[b'labels']
    if X.shape[1] == 32*32*3:
        # Reshape stupid (IMO) original shape...
        X = X.reshape((len(X), 32,32,3), order='F')
        X = np.rot90(X, k=3, axes=(1,2))
    assert len(X) == len(y)
    return X, y

def save_batch(batch_path, X, y):
    batch = {
        b'data': X,
        b'labels': y,
    }
    batch_dir = os.path.dirname(batch_path)
    if not os.path.isdir(batch_dir):
        os.makedirs(batch_dir)
    with open(batch_path, 'wb') as f:
        pickle.dump(batch, f)

def main(args):
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

    zuar_batch(
        os.path.join(CIFAR_10_DIR, 'test_batch'),
        os.path.join(CIFAR_10_ZUADO_DIR, 'test_batch'))


class BatchVisualizer(object):
    def __init__(self, X, y, n_rows, n_cols):
        self.fig, self.axes = plt.subplots(n_rows, n_cols)
        self.page = 0
        self.X = X
        self.y = y
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.draw()

    def show(self):
        plt.show()

    def draw(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                n = self.page*self.n_cols*self.n_rows + i*self.n_cols+j
                self.axes[i,j].imshow(self.X[n])
                self.axes[i,j].axis('off')
        self.fig.canvas.flush_events()
        plt.gcf().canvas.draw_idle()

    def onclick(self, event):
        self.page += 1
        self.draw()
        print('{} click: button={}, x={}, y={}, xdata={}, ydata={}'.format(
            'double' if event.dblclick else 'single',
            event.button, event.x, event.y, event.xdata, event.ydata))

def inspect(args):
    X, y = load_batch(args.batch_path)
    bv = BatchVisualizer(X, y, 2, 5)
    bv.show()


if __name__ == '__main__':
    # Arguments and options
    parser = argparse.ArgumentParser(description="""
        Downloads CIFAR-10 and create a version of the dataset,
        where the images are rotated and translated.""")

    subparsers = parser.add_subparsers(help='Available commands.', dest='command')
    build_parser = subparsers.add_parser('build', help='Build the CIFAR-10 zuado dataset.')

    inspect_parser = subparsers.add_parser('inspect', help='Inspect batch.')
    inspect_parser.add_argument('batch_path')

    args = parser.parse_args()
    if not args.command or args.command == 'build':
        main(args)
    elif args.command == 'inspect':
        inspect(args)
