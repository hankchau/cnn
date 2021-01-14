import os
import data
import shutil
import numpy as np
import illustrate as ill


def plot_data_index():
    path = 'example/sample_output/plot_data_index/'
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

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


def run_contrast_data():
    '''Plots some csv data sample files.'''
    outpath = 'example/sample_output/run_contrast_data/'
    if os.path.isdir(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=True)

    # check data
    fpath1 = 'csv_data/Basement/2020-11-21/172102_000000.csv'
    fpath2 = 'csv_data/Basement/2020-11-24/090936_111111.csv'

    mat1 = data.read_csv(fpath1)
    mat2 = data.read_csv(fpath2)

    # compare '111111' and '000000'
    titles = ['111111', '000000']
    ill.render(mat2, os.path.join(outpath, 'L_1.png'))
    ill.render(mat1, os.path.join(outpath, 'L_0.png'))
    ill.contrast([mat2, mat1], os.path.join(outpath, 'gray_0_1.png'),
               'Grayscale before denoising', titles, cmap='gray')
    ill.contrast([mat2, mat1], os.path.join(outpath, 'rainbow_0_1.png'),
               'Heatmap before denoising', titles, cmap='rainbow')

    # denoise by subtraction
    denoise = mat2 - mat1
    titles = ['denoise', 'abs(denoise)']
    ill.render(denoise, os.path.join(outpath, 'L_subtract.png'))
    ill.contrast([denoise, np.abs(denoise)], os.path.join(outpath, 'gray_subtract.png'),
               'Normalization with Subtraction', titles, cmap='gray')
    ill.contrast([denoise, np.abs(denoise)], os.path.join(outpath, 'rainbow_subtract.png'),
               'Normalization with Subtraction', titles, cmap='rainbow')

    # denoise by cropping image
    c_mat2 = mat2[27:,:]
    titles = ['before cropping', 'after cropping']
    ill.render(c_mat2, os.path.join(outpath, 'crop_L_1.png'))
    ill.contrast([mat2, c_mat2], os.path.join(outpath, 'gray_crop.png'),
               'Normalization after cropping', titles, cmap='gray')
    ill.contrast([mat2, c_mat2], os.path.join(outpath, 'rainbow_crop.png'),
               'Normalization after cropping', titles, cmap='rainbow')


if __name__ == '__main__':
    data.fill_data_index('data/areaMask.csv')
    plot_data_index()
    run_contrast_data()
