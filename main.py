import data
import os
import glob
import cnn
import tensorflow as tf


def main(host, data_dir):
    # scrape data from server
    # Basement
    addr = host
    area = '/Basement/'
    start_date = '2021-01-08'
    # data.get_data(addr, area , data_dir, start_date)

    # OffRoad
    addr = host
    area = '/OffRoad/'
    off_outpath = os.path.join(data_dir, 'OffRoad/')
    #data.get_data(addr, area, data_dir)

    # ParaParking

    # data preprocessing
    outpath = '/Volumes/Transcend/output/'
    # outpath = 'output/'
    data.fill_data_index('data/areaMask.csv')
    # fpattern = 'csv_data/Basement'
    fpath = '/Volumes/Transcend/csv_data/Basement/'
    # class_weights = data.find_label_distr(fpath)

    # produce Heatmap and ROI
    base_outpath = os.path.join(data_dir, 'Basement/')
    hm_dir = os.path.join(outpath, 'heatmaps', 'Basement')
    data.generate_heatmaps(base_outpath, hm_dir, start_date)
    roi_dir = os.path.join(outpath, 'ROIs', 'Basement')
    data.generate_ROIs(hm_dir, roi_dir, start_date)

    hm_dir = os.path.join(outpath, 'heatmaps', 'OffRoad')
    # data.generate_heatmaps(off_outpath, hm_dir)
    roi_dir = os.path.join(outpath, 'ROIs', 'OffRoad')
    # data.generate_ROIs(hm_dir, roi_dir)

    # Training
    '''
    model = cnn.CNN(outpath='saved_files/')
    # model.visualize()
    model.compile()

    fpaths = glob.glob(os.path.join(roi_dir, '**/**/*.png'), recursive=True)
    dist = {'train': 0.8, 'val': 0.2}
    train, val = data.split(fpaths, dist)
    # model.fit(train, val, class_weights=class_weights, save_model=True)
    model.fit(train, val, save_model=True)
    model.plot_accuracy()
    model.plot_loss()

    # convert to tf-lite model
    saved_model = 'saved_files/cnn_model/model'
    outpath = 'saved_files/cnn_model/cnn.tflite'
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    tflite_model = converter.convert()
    with open(outpath, 'wb') as f:
        f.write(tflite_model)
    '''

if __name__ == '__main__':
    host = 'http://crs.comm.yzu.edu.tw:8888'
    # data_dir = 'csv_data/'
    data_dir = '/Volumes/Transcend/csv_data/'

    main(host, data_dir)
