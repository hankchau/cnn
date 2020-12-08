import numpy as np

from PIL import Image, ImageOps


def get_labels(fname):
    labels = fname[-10:-4]

    return list(labels)


def get_time(fname):
    return 0


def read_csv(fpath):
    matrix = np.genfromtxt(fpath, delimiter=',')

    return matrix


def render(matrix, mode='grayscale'):
    max = np.max(matrix)
    print('max = ' + str(max))
    #matrix /= max
    #max = np.max(matrix)
    print('new max = ' + str(max))
    im = Image.fromarray(matrix).convert('L')

    if mode is 'grayscale':
        #return ImageOps.grayscale(im)
        return im

    elif mode is 'heatmap':
        return ImageOps.colorize(im, black='blue', white='red')
