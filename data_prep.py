import csv
import glob
import numpy as np
import pandas
import columnize

from data.data_index import *
from data.data_utils import read_csv, render, get_labels
from PIL import Image, ImageOps


def main():
    fpath = '../csv_data/Basement/2020-11-21/172102_000000.csv'
    print(get_labels(fpath))
    mat = read_csv(fpath)
    # im = render(mat, 'heatmap')
    im = render(mat)
    im.show()

    fpath = '../csv_data/Basement/2020-11-23/164654_111101.csv'
    print(get_labels(fpath))
    mat = read_csv(fpath)
    im = render(mat)
    im.show()

    return 0


if __name__ == '__main__':
    main()
