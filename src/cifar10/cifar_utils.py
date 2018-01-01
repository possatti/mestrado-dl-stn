from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import sys

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
    n_images = len(X)
    Xz = np.empty(shape=(n_images, 64,64,3), dtype=X.dtype)

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