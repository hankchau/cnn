import os
import shutil
import glob
import data
import illustrate as ill


def test_crop_image():
    fpath1 = 'csv_data/Basement/2020-12-01/133847_111111.csv'
    fpath2 = 'csv_data/Basement/2020-12-01/000006_000000.csv'
    hmpath1 = 'example/sample_output/heatmaps/sample_111111.png'
    hmpath2 = 'example/sample_output/heatmaps/sample_000000.png'
    outpath1 = 'example/sample_output/ROIs/sample_111111.png'
    outpath2 = 'example/sample_output/ROIs/sample_000000.png'
    ill.plot_heatmaps([fpath1, fpath2], [hmpath1, hmpath2], roi_boxes=True)
    ill.plot_ROIs([hmpath1, hmpath2], [outpath1, outpath2])


def test_stack_training_data():
    roi_dir = 'example/sample_output/ROIs/'
    x, y = data.load_training_data(roi_dir, save=True)
    print(x.shape)
    print(y.shape)
    x, y = data.load_training_data(saved_x='saved_files/x_train.npy',
                                   saved_y='saved_files/y_train.npy')
    print(x.shape)
    print(y.shape)