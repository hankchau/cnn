import os
import numpy as np
import illustrate
import itertools
import glob
import datetime
import data.data_utils as du
from random import shuffle


def get_data(addr, area, outpath, start_date=None):
    os.makedirs(outpath, exist_ok=True)
    print('scraping data from host ' + addr)
    print('this will take a few hours ...')
    du.download_data(addr, area, outpath, start_date)
    print('finished with scraping data')


def find_average(file_pattern):
    fpaths = glob.glob(file_pattern)
    if len(fpaths) == 0:
        return np.array([])

    mat = np.zeros((64,48))
    num = 0
    for f in fpaths:
        mat += du.read_csv(f)
        num += 1

    return mat / num


def find_label_distr(fpath, outpath=None):
    if outpath is None:
        outpath = os.getcwd()
    if not os.path.isdir(outpath):
        os.makedirs(outpath, exist_ok=True)
    outpath = os.path.join(outpath, 'label_distributions.txt')
    # find all possible label states
    labels = []
    bins = itertools.product('01', repeat=5)
    for b in bins:
        labels.append(''.join(b))

    total = 0
    class_weights = {}
    with open(outpath, 'w+') as f:
        f.write('label  : frequency\n')
        for i in range(len(labels)):
            fpattern =  os.path.join(fpath, '**/*_' + labels[i] + '*.csv')
            ls = glob.glob(fpattern)
            f.write(labels[i] + ' : %i\n' % len(ls))
            total += len(ls)
            class_weights[i] = len(ls)
    for c in class_weights:
        class_weights[c] = total / (32 * class_weights[c])

    return class_weights


def generate_heatmaps(csvpath, outpath, start_date=None):
    os.makedirs(outpath, exist_ok=True)

    fpaths = glob.glob(os.path.join(csvpath, '**/*.csv'), recursive=True)
    outpaths = []
    d1 = None
    if not start_date is None:
        d1 = du.get_datetime(start_date)
    for f in fpaths:
        id = f.rfind('/')
        date = f[:id]
        date = date[date.rfind('/')+1:]
        d2 = du.get_datetime(date)
        if d1 and d1 > d2:
            outpaths.append('')
            continue

        fname = f[id+1:f.rfind('.')]
        fname += '.png'
        fdir = os.path.join(outpath, date)
        if not os.path.isdir(fdir):
            os.makedirs(fdir, exist_ok=True)

        outpaths.append(os.path.join(fdir, fname))

    illustrate.plot_heatmaps(fpaths, outpaths, roi_boxes=False)


def generate_ROIs(hmpath, outpath, start_date=None):
    os.makedirs(outpath, exist_ok=True)

    fpaths = glob.glob(os.path.join(hmpath, '**/*.png'), recursive=True)
    outpaths = []

    d1 = None
    if not start_date is None:
        d1 = du.get_datetime(start_date)
    for f in fpaths:
        f = f.strip(hmpath)
        id = f.rfind('/')
        date = f[:id]
        date = date[date.rfind('/')+1:]
        d2 = du.get_datetime(date)
        if d1 and d1 > d2:
            outpaths.append('')
            continue

        fname = f[id+1:]
        fdir = os.path.join(outpath, date)
        if not os.path.isdir(fdir):
            os.makedirs(fdir, exist_ok=True)

        outpaths.append(os.path.join(fdir, fname))

    illustrate.plot_ROIs(fpaths, outpaths)


def split(fpaths, dist):
    shuffle(fpaths)
    n = len(fpaths)
    id = int(n * dist['train'])
    train = fpaths[:id]
    val = fpaths[id:]

    return train, val


def load_batch(fpaths, batch_size, save_dir=None):
    # find all possible label states
    labels = []
    bins = itertools.product('01', repeat=5)
    for b in bins:
        labels.append(''.join(b))

    shape = du.read_image(fpaths[0]).shape
    x = np.ndarray(shape=(batch_size, shape[0], shape[1], shape[2]))
    y = np.ndarray(shape=(batch_size,len(labels)))

    for i in range(batch_size):
        mat = du.read_image(fpaths[i])
        label = du.get_labels(fpaths[i])
        x[i,:,:,:] = mat / 255
        id = labels.index(label[:-1])
        label = [0] * len(labels)
        label[id] = 1
        y[i,:] = label

    return x, y
