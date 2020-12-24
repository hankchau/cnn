import os
import glob
import data
import shutil
import numpy as np

from illustrate import render, contrast, plot_heatmap


def run_all_tests():
    test_prep()


def test_prep():
    data.fill_data_index('data/areaMask.csv')


def plot_data_index():
    path = 'example/sample_output/'
    if not os.path.isdir(path):
        os.mkdir(path)

    slots = [53, 54, 55, 56, 57, 58, -1, -2, -3, -4]

    # define padded matrix
    h, w = 64, 48
    mat = np.zeros((h,w), dtype=int)
    mat = np.pad(mat, ((3,3), (37,37)), mode='constant', constant_values=(0,))

    x, y = data.get_transform_index()
    # translate x,y coord to 1st quadrant
    x += abs(x.min())
    #y += abs(y.min())

    roi_dict = {}
    dupes = 0
    for s in slots:
        num_ocpd = 0
        row_id = data.data_index[str(s)].pixels[0]
        col_id = data.data_index[str(s)].pixels[1]
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
    np.savetxt(os.path.join(path, 'Pixel_Distr.txt'),
               np.flipud(mat), fmt='%2.d', header=header)

    print('Generating area distributions of ROIs and other zones ...')
    print('"Pixel_Distr.txt" can be found under example/sample_output')


def plot_sample_heatmap():
    path = 'example/sample_output/'
    if not os.path.isdir(path):
        os.mkdir(path)
    fpaths = glob.glob('csv_data/Basement/**/*_000001.csv')
    fpath = fpaths[129]
    #fpath = 'csv_data/Basement/2020-12-08/084739_000100.csv'
    matf = data.read_csv(fpath)
    plot_heatmap(matf, os.path.join('matf_Heatmap_000000.png'))

    file_pattern = 'csv_data/Basement/**/*_000001.csv'
    # mat0 = data.find_average(file_pattern)
    # np.savetxt("avg_000000.csv", mat0, delimiter=",")
    mat0 = data.read_csv('avg_000000.csv')
    mat1 = data.read_csv('csv_data/Basement/2020-11-23/000012_000000.csv')
    mat2 = data.read_csv('csv_data/Basement/2020-11-24/084934_111110.csv')
    mat3 = data.read_csv('csv_data/Basement/2020-11-24/090936_111111.csv')
    dn = matf - mat0
    dn = np.abs(dn)
    #mat0 = 150 * np.random.random_sample((64, 48)) + 50
    plot_heatmap(mat0, os.path.join('Heatmap_Avg_000000.png'))
    plot_heatmap(mat1, os.path.join('Heatmap_000000.png'))
    plot_heatmap(mat2, os.path.join('Heatmap_111110.png'))
    plot_heatmap(mat3, os.path.join('Heatmap_111111.png'))
    plot_heatmap(dn, os.path.join('Heatmap_Denoise_111111.png'))


def run_example0():
    '''Plots some csv data sample files.'''
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


def run_example1():
    path = 'example/sample_output/'
    if not os.path.isdir(path):
        os.mkdir(path)

    mat = data.read_csv('avg_000000.csv')
    plot_heatmap(mat, os.path.join(path, 'Heatmap_Avg_000000.png'))
    mat = data.find_average('csv_data/Basement/**/*_000001.csv')
    plot_heatmap(mat, os.path.join(path, 'Heatmap_Avg_000001.png'))
    mat = data.find_average('csv_data/Basement/**/*_000010.csv')
    plot_heatmap(mat, os.path.join(path, 'Heatmap_Avg_000010.png'))
    mat = data.find_average('csv_data/Basement/**/*_000100.csv')
    plot_heatmap(mat, os.path.join(path, 'Heatmap_Avg_000100.png'))
    mat = data.find_average('csv_data/Basement/**/*_001000.csv')
    plot_heatmap(mat, os.path.join(path, 'Heatmap_Avg_001000.png'))
    mat = data.find_average('csv_data/Basement/**/*_010000.csv')
    plot_heatmap(mat, os.path.join(path, 'Heatmap_Avg_010000.png'))
    mat = data.find_average('csv_data/Basement/**/*_100000.csv')
    plot_heatmap(mat, os.path.join(path, 'Heatmap_Avg_100000.png'))
    mat = data.find_average('csv_data/Basement/**/*_111111.csv')
    plot_heatmap(mat, os.path.join(path, 'Heatmap_Avg_111111.png'))


