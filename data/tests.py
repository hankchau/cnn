import numpy as np
from data.data_index import *
from data import read_csv


def run_all_tests():
    check_data_index()
    sample_data()


def check_data_index():
    print('53, 54, 55, 56, 57, 58 : ')
    print(len(s53), len(s54), len(s54), len(s55), len(s56), len(s57), len(s58))

    height = 64
    width = 48
    mat = np.zeros((height, width), dtype=int)
    indices = [53, 54, 55, 56, 57, 58, 1, 2, 3, 4]
    slots = [s53, s54, s55, s56, s57, s58, A, B, C, D]

    num_ocpd = 0    # number of pixels occupied
    index = 0
    for s in slots:
        for h, w in s:
            mat[w, h] = indices[index]
            num_ocpd += 1
        index += 1

    header = 'ROI: 53 = slot 53, etc.\nA = 1, B = 2, C = 3, D = 4, 0 = None\n'
    header += 'Number of pixels covered:\n53: %i, 54: %i, 55: %i, 56: %i, 57: %i, 58: %i\n' % (
        len(s53), len(s54), len(s55), len(s56), len(s57), len(s58)
    )
    header += '1: %i, 2: %i, 3: %i, 4: %i, 0: %i\n\n' % (
        len(A), len(B), len(C), len(D), 64*48-num_ocpd)

    np.savetxt('data/ROI_and_Area_Distributions.txt', mat, fmt='%2.d', header=header)
    np.savetxt('data/(BottomRadar)ROI_and_Area_Distributions.txt', np.flipud(mat), fmt='%2.d', header=header)

    print('Generating area distributions of ROIs and other zones ...')
    print('"ROI_and_Area_Distributions.txt" can be found under data/')
    print('A user-friendly version with radar at the bottom has been generated too')


def sample_data():
    # check data
    fpath1 = 'csv_data/Basement/2020-11-21/172102_000000.csv'
    fpath2 = 'csv_data/Basement/2020-11-25/100419_111111.csv'
    fpath3 = 'csv_data/Basement/2020-11-24/090936_111111.csv'

    mat1 = read_csv(fpath1)
    mat2 = read_csv(fpath2)
    mat3 = read_csv(fpath3)


    mat1 = mat1[27:,:]
    mat2 = mat2[27:,:]
    mat3 = mat3[27:,:]

    denoise = mat2 - mat1
    abs_denoise = np.abs(mat2 - mat1)
