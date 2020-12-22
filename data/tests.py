import numpy as np
import scipy as spi
import os
import shutil

import data
from illustrate import render, contrast, plot_heatmap


def run_all_tests():
    test_prep()
    run_example()
    data_prep()


def test_prep():
    data.fill_data_index('data/data_index.csv')


def plot_data_index():
    slots = [53, 54, 55, 56, 57, 58, -1, -2, -3, -4]

    # define padded matrix
    h, w = 64, 48
    mat = np.zeros((h,w), dtype=int)
    #mat = np.pad(mat, ((0,0), (25,25)), mode='constant', constant_values=(0,))
    mat = np.pad(mat, ((3,3), (37,37)), mode='constant', constant_values=(0,))

    x, y = data.get_transform_index()
    # translate x,y coord to 1st quadrant
    x += abs(x.min())
    #y += abs(y.min())

    roi_dict = {}
    dupes = 0
    for s in slots:
        num_ocpd = 0
        row_id = data.data_index[str(s)][0]
        col_id = data.data_index[str(s)][1]
        for i in range(len(row_id)):
            n = row_id[i]
            m = col_id[i]
            new_n = int(round(y[n, m]))
            new_m = int(round(x[n, m]))

            if mat[new_n, new_m] != 0:
                dupes += 1
                continue

            mat[new_n, new_m] = s
            num_ocpd += 1
        roi_id = str(s)
        roi_dict[roi_id] = num_ocpd

    header = 'Total Overlapped Pixels: %i\nNumber of pixels covered:\n' % (dupes)
    for id in roi_dict:
        header += (id + ': %i, ' % (roi_dict[id]))
    header += '\n'
    np.savetxt('Post_ROI_and_Area_Distributions.txt', mat, fmt='%2.d', header=header)
    np.savetxt('(BottomRadar)Post_ROI_and_Area_Distributions.txt',
               np.flipud(mat), fmt='%2.d', header=header)

    print('Generating area distributions of ROIs and other zones ...')
    print('"Post_ROI_and_Area_Distributions.txt" can be found under data/')
    print('A user-friendly version with radar at the bottom has been generated too')


def plot_sample_heatmap():
    file_pattern = 'csv_data/Basement/**/*_000000.csv'
    #mat0 = data.find_average(file_pattern)
    #np.savetxt("avg_000000.csv", mat0, delimiter=",")
    mat0 = data.read_csv('avg_000000.csv')

    fpath1 = 'csv_data/Basement/2020-11-23/000012_000000.csv'
    fpath2 = 'csv_data/Basement/2020-11-24/084934_111110.csv'
    fpath3 = 'csv_data/Basement/2020-11-24/090936_111111.csv'
    mat1 = data.read_csv(fpath1)
    mat2 = data.read_csv(fpath2)
    mat3 = data.read_csv(fpath3)

    #mat1[mat1 > 2500.0] = 0
    mat1[:30,] = np.median(mat1)
    mat2[:30,] = np.median(mat2)
    mat3[:30,] = np.median(mat3)
    dn = mat3 - mat0
    #dn = np.abs(dn)
    plot_heatmap(mat0, 'Heatmap_Avg_000000.png')
    plot_heatmap(mat1, 'Heatmap_000000.png')
    plot_heatmap(mat2, 'Heatmap_111110.png')
    plot_heatmap(mat3, 'Heatmap_111111.png')
    plot_heatmap(dn, 'Heatmap_Denoise_111111.png')
    #plot_heatmap(mat2-mat1)


def run_example():
    outpath = 'example/sample_output/'
    if os.path.isdir(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    # check data
    fpath1 = 'csv_data/Basement/2020-11-21/172102_000000.csv'
    fpath2 = 'csv_data/Basement/2020-11-24/090936_111111.csv'

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
