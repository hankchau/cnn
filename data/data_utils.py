import os
import numpy as np
import requests
import data
from bs4 import BeautifulSoup


def get_labels(fname):
    labels = list(fname[-10:-4])
    labels = np.array(labels)

    return labels.astype(float)


def get_transform_index():
    # range_scale = 54.16(area), 69.28(fit)
    range_bins, range_scale = 64 - 1, 12.8
    angle_bins, angle_width = 48 - 1, 60.0

    # initialize mat index arrays
    n = np.arange(range_bins + 1).reshape((1,-1))
    m = np.arange(angle_bins + 1).reshape((1,-1))

    # transform to sector coord
    r = range_scale/range_bins * n
    beta = 2 * angle_width/angle_bins     # angle ranges from -60 ~ 60 degrees
    t = -angle_width + (beta * m)
    t = np.radians(t)

    x = np.matmul(r.T, np.sin(t))
    y = np.matmul(r.T, np.cos(t))

    return x, y


def fill_data_index(fpath):
    mat = np.genfromtxt(fpath, delimiter=',')

    slots = [53, 54, 55, 56, 57, 58]
    corners = [
        [(47,45), (63,38), (63,34), (40,41)],
        [(35,37), (62,31), (60,27), (31,30)],
        [(31,29), (59,26), (59,22), (32,20)],
        [(31,18), (59,21), (61,17), (36,12)],
        [(36,11), (62,16), (63,12), (43,6)],
        [(45,6), (63,11), (63,6), (53,2)]
    ]

    for i in range(len(slots)):
        id = slots[i]
        data.data_index[str(id)].pixels = np.where(mat == id)
        data.data_index[str(id)].set_corners(
            corners[i]
        )


def read_csv(fpath):
    return np.genfromtxt(fpath, delimiter=',')


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
