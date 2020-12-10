import os
import numpy as np
import requests

from bs4 import BeautifulSoup


def get_labels(fname):
    labels = fname[-10:-4]

    return list(labels)


def get_time(fname):
    return 0


def norm(mat):
    max, min = np.max(mat), np.min(mat)
    mat = (mat - min) * (255/(max-min))

    return mat

def norm2(mat):
    mat /= np.max(mat)

    return mat


def read_csv(fpath):
    matrix = np.genfromtxt(fpath, delimiter=',')

    return matrix


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
    with requests.Session() as sess:
        data = []

        for l in csv_links:
            r = sess.get(l)
            save_csv(l, r.content, outpath)


def save_csv(link, content, outpath):
    fname = link[-17:]
    date = link[-28:-17]
    fpath = os.path.join(outpath, date)

    if not os.path.isdir(fpath):
        os.mkdir(fpath)

    fpath = os.path.join(fpath, fname)
    with open(fpath, 'wb') as csv_file:
        csv_file.write(content)
