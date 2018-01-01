#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.callbacks import TensorBoard
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from spatial_transformer import SpatialTransformer

import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse
import random
import sys
import re
import os

from cifar_utils import load_batch, zuar_batch, batch_preprocessing, BatchVisualizer


def create_baseline_model(input_shape=(64, 64, 3), output_dim=10):
    model = Sequential()
    # model.add(Input(shape=input_shape)) # Does this work?
    model.add(Conv2D(48, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(96, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(192, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_stn_model(input_shape=(64, 64, 3), output_dim=10):
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
                             output_size=(64,64), input_shape=input_shape))
    model.add(Conv2D(48, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(96, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(192, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def create_timestamp():
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

MODELS = {
    'baseline': create_baseline_model,
    'stn': create_stn_model,
}

random.seed(7)
np.random.seed(7)

def train(args):
    print('Loading dataset from: {}'.format(args.cifar10), file=sys.stderr)

    X_b1, y_b1 = load_batch(os.path.join(args.cifar10, 'data_batch_1'))
    X_b2, y_b2 = load_batch(os.path.join(args.cifar10, 'data_batch_2'))
    X_b3, y_b3 = load_batch(os.path.join(args.cifar10, 'data_batch_3'))
    X_b4, y_b4 = load_batch(os.path.join(args.cifar10, 'data_batch_4'))
    X_b5, y_b5 = load_batch(os.path.join(args.cifar10, 'data_batch_5'))

    X = np.concatenate([X_b1, X_b2, X_b3, X_b4, X_b5], axis=0)
    y = np.concatenate([y_b1, y_b2, y_b3, y_b4, y_b5], axis=0)

    X, Y = batch_preprocessing(X, y)

    right_now = create_timestamp()
    for model_name, create_fn in MODELS.items():
        if model_name in args.allowed_models:
            tb_logdir = os.path.join(args.tensorboard_dir, model_name)
            model_def_path = os.path.join(args.trained_dir, '{}_{}.json'.format(model_name, right_now))
            model_weights_path = os.path.join(args.trained_dir, '{}_{}.hd5'.format(model_name, right_now))

            print('Creating {} model...'.format(model_name))
            model = create_fn(input_shape=(32,32,3), output_dim=10)
            print('Training {} model...'.format(model_name))
            model.fit(X, Y, validation_split=0.1, epochs=args.epochs,
                batch_size=128, callbacks=[TensorBoard(log_dir=tb_logdir)])
            save_model(model, model_def_path, model_weights_path)

def evaluate(args):
    print('Loading test data from: {}'.format(args.cifar10), file=sys.stderr)
    X, y = load_batch(os.path.join(args.cifar10, 'test_batch'))
    X, Y = batch_preprocessing(X, y)

    for model_name, create_fn in MODELS.items():
        if model_name in args.allowed_models:
            # Find last trained model.
            weights_filepath = None
            for filename in os.listdir(args.trained_dir):
                if re.match(r'{}_.*\.hd5'.format(model_name), filename):
                    weights_filepath = os.path.join(args.trained_dir, filename)

            if weights_filepath:
                print('Creating {} model...'.format(model_name))
                model = create_fn(input_shape=(32,32,3), output_dim=10)
                print('Loading {} weights...'.format(model_name))
                model.load_weights(weights_filepath)
                print('Evaluating {}...'.format(model_name))
                loss_and_metrics = model.evaluate(X, Y, batch_size=128)
                print('\nMetrics on test data using {}:'.format(model_name))
                for metric_name, metric_value in zip(model.metrics_names, loss_and_metrics):
                    print(' - {}: {}'.format(metric_name, metric_value))
                print()
            else:
                print('Could not find trained model for "{}". Skipping.'.format(model_name))

def visualize(args):
    pass_through_st = K.function([stn.input], [stn.layers[0].output])

    samples = X_test[:25]
    samples_transformed = pass_through_st([samples.astype('float32')])

    # Input images.
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(np.squeeze(samples[i]), cmap='gray')
        plt.title('Label: {}'.format(np.argmax(Y_test[i])))
        plt.axis('off')
    plt.show()

    # Output from STN.
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(np.squeeze(samples_transformed[0][i]), cmap='gray')
        plt.title('Label: {}'.format(np.argmax(Y_test[i])))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Defaults.
    default_cifar10_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cifar-10-batches-py'))
    default_cifar10_zuado_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cifar-10-zuado'))
    default_trained_models_dir = os.path.join(os.path.dirname(__file__), 'trained')
    default_tensorboard_dir = os.path.join(os.path.dirname(__file__), 'logs')

    # Arguments and options.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cifar10', default=default_cifar10_dir)
    parser.add_argument('--cifar10-zuado', default=default_cifar10_zuado_dir)
    parser.add_argument('--tensorboard-dir', default=default_tensorboard_dir)
    parser.add_argument('--trained-dir', default=default_trained_models_dir,
        help='Directory where the trained models will be saves or loaded from.')
    parser.add_argument('--only', action='append', choices=MODELS, help='Only run these models.')
    parser.add_argument('--skip', action='append', choices=MODELS, help='Skip these models.')

    # Commands.
    subparsers = parser.add_subparsers(help='Available commands.', dest='command')
    train_parser = subparsers.add_parser('train', help='Train models.')
    train_parser.add_argument('--epochs', type=int, default=10)
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate models on test data.')
    visualize_parser = subparsers.add_parser('visualize', help='Visualize transformation through the STN.')

    # Parse arguments.
    args = parser.parse_args()

    # Find which models should be run.
    args.allowed_models = MODELS.keys()
    if args.skip:
        args.allowed_models = list(filter(lambda name: name not in args.skip, args.allowed_models))
    if args.only:
        args.allowed_models = list(filter(lambda name: name in args.only, args.allowed_models))

    # Run command.
    if not args.command:
        parser.print_help(file=sys.stderr)
    elif args.command == 'train': train(args)
    elif args.command == 'evaluate': evaluate(args)
    elif args.command == 'visualize': visualize(args)
