import data
import os
import glob
import cnn
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

    # Training
    # model = cnn.CNN(outpath='saved_files/')
    # model.visualize()
    # model.compile()

    fpaths = glob.glob(os.path.join(roi_dir, '**/**/*.png'), recursive=True)
    dist = {'train': 0.8, 'val': 0.2}
    train, val = data.split(fpaths, dist)
    # model.fit(train, val, save_model=True)
    # model.plot_accuracy()
    # model.plot_loss()

    # convert to tf-lite model
    saved_model = 'saved_files/cnn_model/model'
    outpath = 'saved_files/cnn_model/cnn.tflite'
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    tflite_model = converter.convert()
    with open(outpath, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    main()
