import os
import numpy as np
import shutil
import illustrate
import itertools
import glob
import data.data_utils as du
from matplotlib import pyplot as plt


def get_data(host, dpath):
    # remove current data
    if os.path.isdir(dpath):
        shutil.rmtree(dpath)
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
    with open('csv_paths.txt', 'w') as file:
        for p in all_data_paths:
            file.write(p)
    print('all data file paths stored in csv_paths.txt')


def download_data(addr, outpath):
    date_links = du.get_links(addr)

    paths = []
    for dl in date_links:
        csv_links = du.get_links(dl)
        path = du.get_csv(csv_links, outpath)
        paths.extend(path)

    return paths


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


def find_label_distr(outpath):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    outpath = os.path.join(outpath, 'label_distributions.txt')
    # find all possible label states
    labels = []
    bins = itertools.product('01', repeat=6)
    for b in bins:
        labels.append(''.join(b))

    with open(outpath, 'w+') as f:
        f.write('label  : frequency\n')

        for l in labels:
            ls = glob.glob('csv_data/**/**/*_' + l + '.csv')
            f.write(l + ' : %i\n' % len(ls))


def generate_heatmaps(csvpath, outpath):
    if os.path.isdir(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=True)

    fpaths = glob.glob(os.path.join(csvpath, '**/**/*.csv'), recursive=True)
    outpaths = []
    for f in fpaths:
        id = f.rfind('/')
        fname = f[id+1:f.rfind('.')]
        fname += '.png'
        fdir = os.path.join(outpath, f[f.find('/')+1:id])
        if not os.path.isdir(fdir):
            os.makedirs(fdir, exist_ok=True)

        outpaths.append(os.path.join(fdir, fname))

    illustrate.plot_heatmaps(fpaths, outpaths, roi_boxes=False)


def generate_ROIs(hmpath, outpath):
    if os.path.isdir(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=True)

    fpaths = glob.glob(os.path.join(hmpath, '**/**/*.png'), recursive=True)
    outpaths = []
    for f in fpaths:
        f = f.strip(hmpath)
        id = f.rfind('/')
        fname = f[id+1:]
        fdir = os.path.join(outpath, f[:id])
        if not os.path.isdir(fdir):
            os.makedirs(fdir, exist_ok=True)

        outpaths.append(os.path.join(fdir, fname))

    illustrate.plot_ROIs(fpaths, outpaths)


def load_training_batch(fpaths, step, batch_size, save_dir=None):
    shape = du.read_image(fpaths[0]).shape
    x = np.ndarray(shape=(batch_size, shape[0], shape[1], shape[2]))
    y = np.ndarray(shape=(batch_size, ㄓ))

    for i in range(batch_size):
        mat = du.read_image(fpaths[i])
        label = du.get_labels(fpaths[i])
        x[i,:,:,:] = mat
        y[i,:] = label

    if save_dir:
        if not os.path.isdir(save_dir):
            print('error: no such directory')
            exit(0)
        np.save(os.path.join(save_dir, 'x_train' + str(step)), x)
        np.save(os.path.join(save_dir, 'y_train' + str(step)), y)
        print('training data matrices saved in \'saved_files/\'')

    return x, y
