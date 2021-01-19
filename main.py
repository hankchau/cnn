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
    #data_dir = '/Volumes/Transcend/csv_data/'
    #data.get_data(host, data_dir)

    # data preprocessing
    #outpath = '/Volumes/Transcend/output/'
    outpath = 'output/'
    data.fill_data_index('data/areaMask.csv')
    #data.find_label_distr(outpath=outpath)

    # extract ROI
    hm_dir = os.path.join(outpath, 'heatmaps')
    #data.generate_heatmaps(data_path, hm_dir)
    roi_dir = os.path.join(outpath, 'ROIs')
    #data.generate_ROIs(hm_dir, roi_dir)


    # build CNN
    model = cnn.CNN()
    # model.visualize()
    model.compile()

    # Train
    fpaths = glob.glob(os.path.join(roi_dir, '**/**/*.png'), recursive=True)
    n = len(fpaths)
    step = 0
    epoch = 0
    batch_size = 512
    val_loss = []
    val_acc = []
    decreasing = 0
    decreasing_threshold = 3

    while True:
        shuffle(fpaths)
        for i in range(0, n, batch_size):
            roi_files = fpaths[i:i+batch_size]
            if i + batch_size >= n:
                roi_files = fpaths[i:]

            x, y = data.load_training_batch(roi_files, step, len(roi_files))
            #x = tf.data.Dataset.from_tensor_slices(x)
            #y = tf.data.Dataset.from_tensor_slices(y)
            print('\nStep: ' + str(step))
            step += 1
            print(len(roi_files))
            dict = model.train_on_batch(x, y, return_dict=True)
            #val_loss.append(dict['loss'])
            #val_acc.append(dict['acc'])
            #model.model.fit(x, y, 512)
            del x, y
        # check early stopping criteria
        if len(val_loss) >= decreasing_threshold:
            if val_loss[-1] > val_loss[-2]:
                if decreasing >= decreasing_threshold:
                    # save model and print metrics
                    model.save('saved_files/')
                    model.save_weights('saved_files/')
                    break
                else:
                    decreasing += 1
            else:
                decreasing = 0
        epoch += 1


if __name__ == '__main__':
    main()
