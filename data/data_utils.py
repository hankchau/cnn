import os
import numpy as np
import requests
import data
from PIL import Image
from bs4 import BeautifulSoup


def get_labels(fname):
    return fname[fname.rfind('_')+1:fname.rfind('.')]


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

    # --- Based on 2020-10-30 measured data
    slots = [53, 54, 55, 56, 57, 58]
    corners = [
        [(5.47578, 5.7336), (7.8109, 5.53696), (7.8109, 12), (5.47578, 12)],
        [(1.52773, 6.00399), (4.0286, 5.83192), (4.0286, 12), (1.52773, 12)],
        [(-0.993809, 6.07773), (1.52773, 6.00399), (1.52773, 12), (-0.993809, 12)],
        [(-3.66002, 6.07773), (-0.993809, 6.07773), (-0.993809, 12), (-3.66002, 12)],
        [(-6.24357, 6.12689), (-3.66002, 6.07773), (-3.66002, 12), (-6.24357, 12)],
        [(-8.88911, 5.95483), (-6.24357, 6.12689), (-6.24357, 12), (-8.88911, 12)]
    ]

    for i in range(len(slots)):
        id = slots[i]
        data.data_index[str(id)].pixels = np.where(mat == id)
        data.data_index[str(id)].set_corners(
            corners[i]
        )


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
            if path:
                paths.append(path)

    return paths


def read_csv(fpath):
    return np.genfromtxt(fpath, delimiter=',')


def save_csv(link, content, outpath):
    id = link.rfind('/')
    fname = link[id+1:]
    if len(fname) < 13:
        return None

    date = link[:id]
    date = date[-10:]
    fpath = os.path.join(outpath, date)
    if not os.path.isdir(fpath):
        os.mkdir(fpath)

    fpath = os.path.join(fpath, fname)
    with open(fpath, 'wb') as csv_file:
        csv_file.write(content)

    return fpath


def read_image(fpath):
    with Image.open(fpath) as im:
        im.convert('RGB')
        im = im.resize((86,256))
        mat = np.array(im)

    return mat
