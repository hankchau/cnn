import os
import numpy as np
import itertools
import glob
from data.data_utils import *


def get_data(host, dpath):
    os.mkdir(dpath)
    all_data_paths = []     # list of all csv paths

    print('scraping data from host ' + host)
    print('this will take a few hours ...')
    # Basement
    addr = host + '/Basement/'
    outpath = os.path.join(dpath + 'Basement/')
    os.mkdir(outpath)
    paths = download_data(addr, outpath)
    all_data_paths.extend(paths)
    print('finished with the Basement data')

    # OffRoad
    # addr = host + '/OffRoad/'
    # outpath = os.path.join(dpath + 'OffRoad/')
    # os.mkdir(outpath)
    # paths = download_data(addr, outpath)
    # all_data_paths.extend(paths)
    # print('finished with the OffRoad data')

    # ParaParking
    # addr = host + '/ParaParking/'
    # outpath = os.path.join(dpath + 'ParaParking/')
    # os.mkdir(outpath)
    # paths = download_data(addr, outpath)
    # all_data_paths.extend(paths)
    # print('finished with the ParaParking data')

    # save all output paths
    with open('csv_paths.txt', 'wb') as file:
        for p in all_data_paths:
            file.write(p)
    print('all data file paths stored in csv_paths.txt')


def download_data(addr, outpath):
    date_links = get_links(addr)

    paths = []
    for dl in date_links:
        csv_links = get_links(dl)
        path = get_csv(csv_links, outpath)
        paths.extend(path)

    return paths


def find_average(file_pattern):
    fpaths = glob.glob(file_pattern)
    if len(fpaths) == 0:
        return np.array([])

    mat = np.zeros((64,48))
    num = 0
    for f in fpaths:
        mat += read_csv(f)
        num += 1

    return mat / num


def find_label_distr():
    # find all possible label states
    labels = []
    bins = itertools.product('01', repeat=6)
    for b in bins:
        labels.append(''.join(b))

    with open('label_distributions.txt', 'w+') as f:
        f.write('label  : frequency\n')

        for l in labels:
            ls = glob.glob('csv_data/Basement/**/*_' + l + '.csv')
            f.write(l + ' : %i\n' % len(ls))


def stack_training_data(data_dir):
    data_dir = os.path.join(data_dir, '**/**/*.csv')
    fpaths = glob.glob(data_dir, recursive=True)

    h, w, n, label_dim = 64, 48, len(fpaths), 6
    # n = 1000
    x = np.empty((h,w,n), dtype=float)
    y = np.empty((label_dim,n), dtype=int)

    for i in range(n):
        fname = fpaths[i]
        x[:,:,i] = read_csv(fname)
        y[:,i] = get_labels(fname)

    return x, y
