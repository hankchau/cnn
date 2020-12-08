import os
import requests
from bs4 import BeautifulSoup


def get_data(addr, outpath):
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


def main():
    host = 'http://crs.comm.yzu.edu.tw:8888'

    addr = host + '/Basement/'
    outpath = '../csv_data/Basement/'
    get_data(addr, outpath)

    # addr = host + '/OffRoad/'
    # outpath = '../csv_data/OffRoad/'
    # get_data(addr, outpath)

    # addr = host + '/ParaParking/'
    # outpath = '../csv_data/ParaParking'
    # get_data(addr, outpath)


if __name__ == '__main__':
    main()