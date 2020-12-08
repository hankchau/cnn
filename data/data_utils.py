import numpy as np
import os
import requests

from bs4 import BeautifulSoup
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


def get_data(host):
    dpath = 'csv_data/'
    os.mkdir(dpath)

    print('scraping data from host ' + host)
    print('this will take a few hours ...')
    # Basement
    addr = host + '/Basement/'
    outpath = os.path.join(dpath + 'Basement/')
    os.mkdir(outpath)
    download_data(addr, outpath)
    print('finished with the Basement data')

    # OffRoad
    # addr = host + '/OffRoad/'
    # outpath = os.path.join(dpath + 'OffRoad/')
    # os.mkdir(outpath)
    # download_data(addr, outpath)
    # print('finished with the OffRoad data')

    # ParaParking
    # addr = host + '/ParaParking/'
    # outpath = os.path.join(dpath + 'ParaParking/')
    # os.mkdir(outpath)
    # download_data(addr, outpath)
    # print('finished with the ParaParking data')


def download_data(addr, outpath):
    date_links = get_links(addr)

    for dl in date_links:
        csv_links = get_links(dl)
        get_csv(csv_links, outpath)


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
