#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

def baseline_model(input_shape=(1600,), output_dim=10):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Conv2D, MaxPooling2D

    model = Sequential()
    model.add(Conv2D(64, (11, 11), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(128, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def stn_model(input_shape=(1600,), output_dim=10):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from spatial_transformer import SpatialTransformer


    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]

    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
    locnet.add(Conv2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Conv2D(20, (5, 5)))
    locnet.add(Flatten())
    locnet.add(Dense(50))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights))

    model = Sequential()
    model.add(SpatialTransformer(localization_net=locnet,
                             output_size=(30,30), input_shape=input_shape))
    model.add(Conv2D(64, (11, 11), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(128, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def main(args):
    print('Loading Cluttered MNIST from: {}'.format(args.dataset), file=sys.stderr)
    mnist_cluttered = np.load(args.dataset)

    X_train = mnist_cluttered['X_train']
    X_valid = mnist_cluttered['X_valid']
    X_test = mnist_cluttered['X_test']
    y_train = mnist_cluttered['y_train']
    y_valid = mnist_cluttered['y_valid']
    y_test = mnist_cluttered['y_test']

    # Reshape images to 40x40.
    X_train = X_train.reshape((len(X_train), 40, 40, 1))
    X_valid = X_valid.reshape((len(X_valid), 40, 40, 1))
    X_test = X_test.reshape((len(X_test), 40, 40, 1))

    # One-hot.
    from keras.utils import to_categorical
    Y_train = to_categorical(y_train, num_classes=10)
    Y_valid = to_categorical(y_valid, num_classes=10)
    Y_test = to_categorical(y_test, num_classes=10)

    # Shuffle.
    X_train, Y_train = shuffle_in_unison(X_train, Y_train)
    X_valid, Y_valid = shuffle_in_unison(X_valid, Y_valid)
    X_test, Y_test = shuffle_in_unison(X_test, Y_test)

    # # Visualize data.
    # subplots_xnumber = 5
    # subplots_number = subplots_xnumber ** 2
    # for page in range(int(1000 / subplots_number)):
    #     print("page:", page + 1, file=sys.stderr)
    #     fig, axes = plt.subplots(subplots_xnumber, subplots_xnumber)
    #     for i in range(subplots_xnumber):
    #         for j in range(subplots_xnumber):
    #             n = page*subplots_number + i*subplots_xnumber+j
    #             axes[i,j].imshow(X_test.reshape(1000, 40, 40)[n,...], cmap='gray')
    #             # axes[i,j].set_title('Label: {}'.format(y_test[n]))
    #             axes[i,j].set_title('{}'.format(Y_test[n]))
    #             axes[i,j].axis('off')
    #     plt.show()
    #     break

    from keras.callbacks import TensorBoard

    print('Training baseline model...')
    baseline = baseline_model(input_shape=(40,40,1), output_dim=10)
    baseline.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=args.epochs,
        batch_size=128, callbacks=[TensorBoard(log_dir='./logs/baseline')])
    loss_and_metrics = baseline.evaluate(X_test, Y_test, batch_size=128)
    print('Metrics on test data:')
    for metric_name, metric_value in zip(baseline.metrics_names, loss_and_metrics):
        print(' - {}: {}'.format(metric_name, metric_value))

        print('Training stn model...')
    stn = stn_model(input_shape=(40,40,1), output_dim=10)
    stn.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=args.epochs,
        batch_size=128, callbacks=[TensorBoard(log_dir='./logs/stn')])
    loss_and_metrics = stn.evaluate(X_test, Y_test, batch_size=128)
    print('Metrics on test data:')
    for metric_name, metric_value in zip(stn.metrics_names, loss_and_metrics):
        print(' - {}: {}'.format(metric_name, metric_value))


if __name__ == '__main__':
    # Defaults.
    default_cluttered_mnist_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mnist_sequence1_sample_5distortions5x5.npz'))

    # Arguments and options.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default=default_cluttered_mnist_path)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)
