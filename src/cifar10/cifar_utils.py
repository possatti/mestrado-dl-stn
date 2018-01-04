import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os

try:
    import cPickle as pickle
except:
    import pickle

def load_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    X = batch[b'data']
    y = batch[b'labels']
    if X.shape[1] == 32*32*3:
        # Change original shape.
        X = X.reshape((len(X), 3,32,32))
        X = np.moveaxis(X, 1, -1)
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

def batch_preprocessing(X, y):
    """Prepare batch for training."""
    from keras.utils import to_categorical
    newX = X.astype('float32') / 255
    onehot = to_categorical(y, num_classes=10)
    return newX, onehot

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
        if (self.page+2)*self.n_rows*n_cols <= len(self.X):
            self.page += 1
            self.draw()
        print('Current page: {}'.format(self.page))

def zuar_batch(X, rotation_range=60, horizontal_flip=True,
    horizontal_translation_range=20, vertical_translation_range=20):
    from PIL import Image
    n_images = len(X)
    original_dtype = X.dtype
    Xz = np.empty(shape=(n_images, 64,64,3), dtype=original_dtype)

    if original_dtype != np.uint8:
        raise ValueError('dtype should be uint8, got dtype {}.'.format(original_dtype))
    if X.shape[1:] != (32,32,3):
        raise ValueError('Shape should be (_,32,32,3), got shape {}.'.format(X.shape))

    for i in range(n_images):
        # Define parameters for transformation.
        angle_deg = random.randint(-int(rotation_range/2), int(rotation_range/2))
        tx = random.randint(-int(horizontal_translation_range/2), int(horizontal_translation_range/2))
        ty = random.randint(-int(vertical_translation_range/2), int(vertical_translation_range/2))
        sx, sy = 1, 1
        do_flip = False
        if horizontal_flip:
            do_flip = random.randint(0,1)

        # Transform image.
        im_m = X[i,...]
        im = Image.fromarray(im_m)
        newim = Image.new(im.mode, (64,64))
        newim.paste(im, box=(16,16))
        if do_flip:
            newim = newim.transpose(Image.FLIP_LEFT_RIGHT)
        newim = newim.rotate(angle_deg)
        newim = newim.transform((64,64), Image.AFFINE, (sx,0,tx, 0,sy,ty))
        newim_m = np.asarray(newim, dtype=np.uint8).reshape(64,64,3)
        Xz[i] = newim_m

    return Xz

def generate_distorted_batches(X, y, batch_size=100, preprocess=True):
    X = zuar_batch(X)
    assert len(X) % batch_size == 0, 'ERR:  Total size should be multiple of batch size!'
    for i in range(0, len(X), batch_size):
        batch_begin = i
        batch_end = i + batch_size
        if preprocess:
            yield batch_preprocessing(X[batch_begin:batch_end], y[batch_begin:batch_end])
        else:
            yield (X[batch_begin:batch_end], y[batch_begin:batch_end])
