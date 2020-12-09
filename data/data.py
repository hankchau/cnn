import os
from data.data_utils import *


def get_data(host, dpath):
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


def stack_training_matrix(data_path):
    return 0


def labels_summary():
    return 0
