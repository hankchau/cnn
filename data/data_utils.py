import os
import numpy as np
import requests
import data
import datetime
from PIL import Image
from bs4 import BeautifulSoup


def get_labels(fname):
    return fname[fname.rfind('_')+1:fname.rfind('.')]


def get_datetime(date):
    list = date.split('-')
    return datetime.datetime(int(list[0]), int(list[1]), int(list[2]))


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


def download_data(host, area, outpath, start_date):
    if not os.path.isdir(outpath):
        print('Error: Please mount Transcend USB before trying again!')
        exit(0)
    os.makedirs(outpath, exist_ok=True)

    d1 = None
    if not start_date is None:
        d1 = get_datetime(start_date)

    r = None
    while r is None or r.status_code != 200:
        try:
            r = requests.get(host + area, timeout=10)
        except requests.exceptions.RequestException:
            print('Failed to Connect. Retrying ...')
            continue
    soup = BeautifulSoup(r.text, 'html.parser')
    date_tags = soup.find_all('a', href=True)
    date_tags.pop(0), date_tags.pop(-1)     # remove '../' and 'estatic' hrefs

    for date in date_tags:
        if not date.text[-1] == '/':
            continue
        d2 = date.text.rstrip('/')
        d2 = get_datetime(d2)
        if d1 and d2 < d1:
            continue

        datepath = os.path.join(outpath, date['href'].strip('/'))
        if not os.path.isdir(datepath):
            os.makedirs(datepath, exist_ok=True)
        r = None
        while r is None or r.status_code != 200:
            try:
                r = requests.get(host + date['href'], timeout=5)
            except requests.exceptions.RequestException:
                print('Failed to Connect. Retrying ...')
                continue
        soup = BeautifulSoup(r.text, 'html.parser')
        jpg_tags = soup.find_all('a', href=True)
        jpg_tags.pop(0), jpg_tags.pop(-1)       # remove '../' and 'estatic' hrefs

        for jpg in jpg_tags:
            url = jpg['href']
            if '.csv' in url:
                req = None
                while req is None or r.status_code != 200:
                    try:
                        req = requests.get(host + url, timeout=5)
                    except requests.exceptions.RequestException:
                        print('Failed to Connect. Retrying ...')
                        continue
                fpath = os.path.join(outpath, url.strip('/'))
                with open(fpath, 'wb') as f:
                    f.write(req.content)


def read_csv(fpath):
    return np.genfromtxt(fpath, delimiter=',')


def read_image(fpath):
    with Image.open(fpath) as im:
        im.convert('RGB')
        im = im.resize((256,86))
        mat = np.array(im)

    return mat
