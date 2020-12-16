import numpy as np
import scipy as spi
import os
import shutil

import data
from illustrate import render, contrast


def run_all_tests():
    check_data_index()
    run_example()
    data_prep()


def check_data_index():
    h, w = 64, 48
    mat = np.zeros((h, w), dtype=int)
    indices = [53, 54, 55, 56, 57, 58, 1, 2, 3, 4]
    slots = [data.s53, data.s54, data.s55, data.s56, data.s57, data.s58,
             data.A, data.B, data.C, data.D]

    num_ocpd = 0    # number of pixels occupied
    index = 0
    for s in slots:
        for m, n in s:
            mat[n, m] = indices[index]
            num_ocpd += 1
        index += 1

    header = 'ROI: 53 = slot 53, etc.\nA = 1, B = 2, C = 3, D = 4, 0 = None\n'
    header += 'Number of pixels covered:\n53: %i, 54: %i, 55: %i, 56: %i, 57: %i, 58: %i\n' % (
        len(data.s53), len(data.s54), len(data.s55), len(data.s56), len(data.s57), len(data.s58)
    )
    header += '1: %i, 2: %i, 3: %i, 4: %i, 0: %i\n\n' % (
        len(data.A), len(data.B), len(data.C), len(data.D), 64*48-num_ocpd)

    np.savetxt('data/ROI_and_Area_Distributions.txt', mat, fmt='%2.d', header=header)
    np.savetxt('data/(BottomRadar)ROI_and_Area_Distributions.txt', np.flipud(mat), fmt='%2.d', header=header)

    print('Generating area distributions of ROIs and other zones ...')
    print('"ROI_and_Area_Distributions.txt" can be found under data/')
    print('A user-friendly version with radar at the bottom has been generated too')


def plot_data_index():
    indices = [53, 54, 55, 56, 57, 58, 1, 2, 3, 4]
    slots = [data.s53, data.s54, data.s55, data.s56, data.s57, data.s58,
             data.A, data.B, data.C, data.D]

    # define padded matrix
    h, w = 64, 48
    mat = np.zeros((h,w), dtype=int)
    mat = np.pad(mat, ((0,0), (0,51)), mode='constant', constant_values=(0,))

    x, y = data.transform()
    # translate x,y coord to 1st quadrant
    x += abs(x.min())
    y += abs(y.min())

    dupes = 0
    index = 0
    for s in slots:
        for m, n in s:
            new_m = int(round(x[m, n]))
            new_n = int(round(y[m, n]))
            if mat[new_n, new_m] is not 0:
                dupes += 1
            mat[new_n, new_m] = indices[index]
        index += 1
    print(str(dupes))
    np.savetxt('data/Post_ROI_and_Area_Distributions.txt', mat, fmt='%2.d')
    np.savetxt('data/(BottomRadar)Post_ROI_and_Area_Distributions.txt', np.flipud(mat), fmt='%2.d')

    print('Generating area distributions of ROIs and other zones ...')
    print('"Post_ROI_and_Area_Distributions.txt" can be found under data/')
    print('A user-friendly version with radar at the bottom has been generated too')


def run_example():
    outpath = 'example/sample_output/'
    if os.path.isdir(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # check data
    fpath1 = 'csv_data/Basement/2020-11-21/172102_000000.csv'
    fpath2 = 'csv_data/Basement/2020-11-24/090936_111111.csv'
    fpath3 = 'csv_data/Basement/2020-11-25/100419_111111.csv'

    mat1 = data.read_csv(fpath1)
    mat2 = data.read_csv(fpath2)

    # compare '111111' and '000000'
    titles = ['111111', '000000']
    render(mat2, os.path.join(outpath, 'L_1.png'))
    render(mat1, os.path.join(outpath, 'L_0.png'))
    contrast([mat2, mat1], os.path.join(outpath, 'gray_0_1.png'),
               'Grayscale before denoising', titles, cmap='gray')
    contrast([mat2, mat1], os.path.join(outpath, 'rainbow_0_1.png'),
               'Heatmap before denoising', titles, cmap='rainbow')

    # denoise by subtraction
    denoise = mat2 - mat1
    titles = ['denoise', 'abs(denoise)']
    render(denoise, os.path.join(outpath, 'L_subtract.png'))
    contrast([denoise, np.abs(denoise)], os.path.join(outpath, 'gray_subtract.png'),
               'Normalization with Subtraction', titles, cmap='gray')
    contrast([denoise, np.abs(denoise)], os.path.join(outpath, 'rainbow_subtract.png'),
               'Normalization with Subtraction', titles, cmap='rainbow')

    # denoise by cropping image
    c_mat2 = mat2[27:,:]
    titles = ['before cropping', 'after cropping']
    render(c_mat2, os.path.join(outpath, 'crop_L_1.png'))
    contrast([mat2, c_mat2], os.path.join(outpath, 'gray_crop.png'),
               'Normalization after cropping', titles, cmap='gray')
    contrast([mat2, c_mat2], os.path.join(outpath, 'rainbow_crop.png'),
               'Normalization after cropping', titles, cmap='rainbow')


def data_prep():
    slots = ['53', '54', '55', '56', '57', '58']
    x, y = data.stack_training_data('csv_data')
    uniq, freq = np.unique(y, return_counts=True, axis=1)
    ones = list(np.sum(y, axis=1))
    n = freq.shape[0]
    with open('labels_distributions.txt', 'w+') as f:
        f.write('Total : ' + str(n))
        for i in range(len(ones)):
            f.write(slots[i] + ' : ' + str(ones[i]) + '\n')

        for i in range(n):
            labels = uniq[:,i]
            labels = list(labels.astype(str))
            labels = ''.join(labels)
            count = freq[i]
            f.write(str(labels) + ' : ' + str(count) + '\n')
