#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from keras import backend as K
from keras.models import Sequential, model_from_json
from keras.callbacks import TensorBoard
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from spatial_transformer import SpatialTransformer

import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse
import sys
import re
import os

def create_baseline_model(input_shape=(1600,), output_dim=10):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_stn_model(input_shape=(1600,), output_dim=10):
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
                             output_size=(40,40), input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
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

def now():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def save_model(model, model_path, weights_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    model_json = model.to_json()
    with open(model_path, 'w') as f:
        f.write(model_json)
    model.save_weights(weights_path)

def load_model(model_path, weights_path):
    json_file = open(model_path, 'r')
    with open(model_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json, {'SpatialTransformer': SpatialTransformer})
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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

    # Shuffle. (Should not be necessary for training, since `fit` already shuffles.)
    np.random.seed(7)
    # X_train, Y_train = shuffle_in_unison(X_train, Y_train)
    # X_valid, Y_valid = shuffle_in_unison(X_valid, Y_valid)
    X_test, Y_test = shuffle_in_unison(X_test, Y_test)


    right_now = now()
    baseline_model_def_path = os.path.join(args.trained_dir, 'baseline_{}.json'.format(right_now))
    baseline_model_wights_path = os.path.join(args.trained_dir, 'baseline_{}.hd5'.format(right_now))
    stn_model_def_path = os.path.join(args.trained_dir, 'stn_{}.json'.format(right_now))
    stn_model_wights_path = os.path.join(args.trained_dir, 'stn_{}.hd5'.format(right_now))

    # Train or load baseline model.
    baseline = create_baseline_model(input_shape=(40,40,1), output_dim=10)
    if args.baseline_weights:
        print('Loading baseline model...')
        baseline.load_weights(args.baseline_weights)
    else:
        print('Training baseline model...')
        baseline.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=args.epochs,
            batch_size=128, callbacks=[TensorBoard(log_dir=os.path.join(args.tensorboard_dir, 'baseline'))])
        save_model(baseline, baseline_model_def_path, baseline_model_wights_path)

    # Train or load STN model.
    stn = create_stn_model(input_shape=(40,40,1), output_dim=10)
    if args.stn_weights:
        print('Loading stn model...')
        stn.load_weights(args.stn_weights)
    else:
        print('Training stn model...')
        stn.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=args.epochs,
            batch_size=128, callbacks=[TensorBoard(log_dir=os.path.join(args.tensorboard_dir, 'stn'))])
        save_model(stn, stn_model_def_path, stn_model_wights_path)

    # Evaluate models.
    if args.command == 'evaluate':
        loss_and_metrics = baseline.evaluate(X_test, Y_test, batch_size=128)
        print('Metrics on test data using baseline:')
        for metric_name, metric_value in zip(baseline.metrics_names, loss_and_metrics):
            print(' - {}: {}'.format(metric_name, metric_value))
        print()

        loss_and_metrics = stn.evaluate(X_test, Y_test, batch_size=128)
        print('Metrics on test data using stn model:')
        for metric_name, metric_value in zip(stn.metrics_names, loss_and_metrics):
            print(' - {}: {}'.format(metric_name, metric_value))
        print()

    # Visualize images passing through the STN.
    if args.command == 'visualize':
        pass_through_st = K.function([stn.input], [stn.layers[0].output])

        rows = args.rows
        cols = args.columns
        num = rows*cols

        samples = X_test[:num*2]
        samples_transformed = pass_through_st([samples.astype('float32')])

        if args.side_by_side:
            for i in range(int(num)):
                plt.subplot(rows, cols*2, i*2+1)
                plt.imshow(np.squeeze(samples[i]), cmap='gray')
                plt.title('{} (O.)'.format(np.argmax(Y_test[i])))
                plt.axis('off')
                plt.subplot(rows, cols*2, i*2+2)
                plt.imshow(np.squeeze(samples_transformed[0][i]), cmap='gray')
                plt.title('{} (T.)'.format(np.argmax(Y_test[i])))
                plt.axis('off')
            plt.show()
        else:
            # Input images.
            for i in range(num):
                plt.subplot(rows, cols, i+1)
                plt.imshow(np.squeeze(samples[i]), cmap='gray')
                plt.title('{}'.format(np.argmax(Y_test[i])))
                plt.axis('off')
            plt.show()

            # Output from STN.
            for i in range(num):
                plt.subplot(rows, cols, i+1)
                plt.imshow(np.squeeze(samples_transformed[0][i]), cmap='gray')
                plt.title('{}'.format(np.argmax(Y_test[i])))
                plt.axis('off')
            plt.show()

if __name__ == '__main__':
    # Defaults.
    default_cluttered_mnist_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mnist_sequence1_sample_5distortions5x5.npz'))
    default_trained_models_dir = os.path.join(os.path.dirname(__file__), 'trained')
    default_tensorboard_dir = os.path.join(os.path.dirname(__file__), 'logs')

    # Arguments and options.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default=default_cluttered_mnist_path)
    parser.add_argument('--baseline-model-def')
    parser.add_argument('--baseline-weights')
    parser.add_argument('--stn-model-def')
    parser.add_argument('--stn-weights')
    parser.add_argument('--tensorboard-dir', default=default_tensorboard_dir)
    parser.add_argument('--trained-dir', default=default_trained_models_dir,
        help='Directory where the trained models will be saves or loaded from.')
    parser.add_argument('--load-last', action='store_true', help='Load the last trained models.')
    parser.add_argument('--epochs', type=int, default=10)

    # Commands.
    subparsers = parser.add_subparsers(help='Available commands.', dest='command')
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate both models (may need training).')
    visualize_parser = subparsers.add_parser('visualize', help='Visualize transformation through the STN.')
    visualize_parser.add_argument('--rows', '-r', type=int, default=5)
    visualize_parser.add_argument('--columns', '-c', type=int, default=5)
    visualize_parser.add_argument('--side-by-side', action='store_true')

    # Parse arguments.
    args = parser.parse_args()
    if not args.command:
        parser.print_help(file=sys.stderr)

    # Find last trained model for each model.
    if args.load_last:
        stn_hd5s = []
        stn_model_defs = []
        baseline_hd5s = []
        baseline_model_defs = []
        for root, dirs, files in os.walk(args.trained_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                if re.match(r'stn_.*\.hd5', filename):
                    stn_hd5s.append(filepath)
                elif re.match(r'stn_.*\.json', filename):
                    stn_model_defs.append(filepath)
                elif re.match(r'baseline_.*\.hd5', filename):
                    baseline_hd5s.append(filepath)
                elif re.match(r'baseline_.*\.json', filename):
                    baseline_model_defs.append(filepath)
        if len(stn_hd5s) > 0 and len(stn_model_defs) > 0:
            args.stn_weights = stn_hd5s[-1]
            args.stn_model_def = stn_model_defs[-1]
        if len(baseline_hd5s) > 0 and len(baseline_model_defs) > 0:
            args.baseline_weights = baseline_hd5s[-1]
            args.baseline_model_def = baseline_model_defs[-1]

    main(args)
