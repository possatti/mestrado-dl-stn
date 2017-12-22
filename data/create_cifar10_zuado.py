#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
import requests
import argparse
import shutil
import json
import sys
import re
import os

print(sys.version, file=sys.stderr)
print('Numpy version:', np.version.version, file=sys.stderr)

CIFAR_10_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_10_TARGZ_FILEPATH = os.path.join(os.path.dirname(__file__), 'cifar-10-python.tar.gz')
CIFAR_10_DIR = os.path.join(os.path.dirname(__file__))

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
        shutil.unpack_archive(CIFAR_10_TARGZ_FILEPATH, CIFAR_10_DIR)
    else:
        print("CIFAR 10 already unpacked to: `{}`.".format(CIFAR_10_DIR), file=sys.stderr)



if __name__ == '__main__':

    # Arguments and options
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    main(args)
