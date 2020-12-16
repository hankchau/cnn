import os
import math
import numpy as np
import requests

from bs4 import BeautifulSoup


def get_labels(fname):
    labels = list(fname[-10:-4])
    labels = np.array(labels)

    return labels.astype(float)


def get_time(fname):
    return 0


def normalize(mat, range):
    max, min = np.max(mat), np.min(mat)
    mat = (mat - min) * (range/(max-min))

    return mat


def transform():
    range_bins, range_scale = 64 - 1, 63
    angle_bins, angle_width = 48 - 1, 60.0

    # initialize mat index arrays
    m = np.arange(range_bins + 1).reshape((1,-1))
    n = np.arange(angle_bins + 1).reshape((1,-1))

    # transform to sector coord
    r = range_scale/range_bins * n
    beta = 2 * angle_width/angle_bins     # angle ranges from 150 - 30 degrees

    t = -angle_width + (beta * m)
    t = np.radians(t)

    x = np.matmul(r.T, np.sin(t))
    y = np.matmul(r.T, np.cos(t))

    return x, y


def read_csv(fpath):
    matrix = np.genfromtxt(fpath, delimiter=',')

    return matrix


def crop_matrix(mat, index, axis=0):
    if axis is 0:
        return mat[:index, :]
    elif axis is 1:
        return mat[:,:index]
    elif axis is 'both':
        try:
            return mat[:index[0], :index[1]]
        except IndexError as e:
            print('setting axis as "both" requires two indices')


def get_links(addr, host='http://crs.comm.yzu.edu.tw:8888'):
    r = requests.get(addr)
    soup = BeautifulSoup(r.text, 'html.parser')
    link_tags = soup.find_all('a', href=True)
    link_tags.pop(0), link_tags.pop(-1)     # remove '../' and 'estatic' hrefs

    links = []
    for tag in link_tags:
        links.append(host + tag['href'])

    return links


def get_csv(csv_links, outpath):
    # get all data in each date
    paths = []
    with requests.Session() as sess:
        for l in csv_links:
            r = sess.get(l)
            path = save_csv(l, r.content, outpath)
            paths.append(path)

    return paths


def save_csv(link, content, outpath):
    fname = link[-17:]
    date = link[-28:-17]
    fpath = os.path.join(outpath, date)

    if not os.path.isdir(fpath):
        os.mkdir(fpath)

    fpath = os.path.join(fpath, fname)
    with open(fpath, 'wb') as csv_file:
        csv_file.write(content)

    return fpath
