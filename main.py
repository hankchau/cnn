import data
import os
import glob
import cnn
from random import shuffle
import tensorflow as tf


def main():
    # scrape data from server
    host = 'http://crs.comm.yzu.edu.tw:8888'
    data_dir = 'csv_data/'
    # data_dir = '/Volumes/Transcend/csv_data/'
    # data.get_data(host, data_dir)

    # data preprocessing
    # outpath = '/Volumes/Transcend/output/'
    outpath = 'output/'
    data.fill_data_index('data/areaMask.csv')
    # data.find_label_distr(outpath=outpath)

    # extract ROI
    hm_dir = os.path.join(outpath, 'heatmaps')
    # data.generate_heatmaps(data_path, hm_dir)
    roi_dir = os.path.join(outpath, 'ROIs')
    # data.generate_ROIs(hm_dir, roi_dir)


    # build CNN
    model = cnn.CNN(outpath='saved_files/')
    # model.visualize()
    model.compile()

    # Train
    fpaths = glob.glob(os.path.join(roi_dir, '**/**/*.png'), recursive=True)
    dist = {'train': 0.8, 'val': 0.2}
    train, val = data.split(fpaths, dist)
    model.fit(train, val, save_model=True)
    model.plot_accuracy()
    model.plot_loss()
    # model.predict(test)


if __name__ == '__main__':
    main()
