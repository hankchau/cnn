#from cnn import CNN
import data
import os
import glob


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
    # model = CNN()
    # model.visualize()

    # Train
    fpaths = glob.glob(os.path.join(roi_dir, '**/**/*.png'), recursive=True)
    n = len(fpaths)
    step = 0
    batch_size = 30000
    for i in range(0, n, batch_size):
        roi_files = fpaths[i:i+batch_size]
        if i + batch_size >= n:
            roi_files = fpaths[i:]

        x, y = data.load_training_batch(roi_files, step, len(roi_files))
        print('\nStep: ' + str(step))
        step += 1
        print(len(roi_files))
        print(x.shape)
        print(y.shape)
        del x, y


if __name__ == '__main__':
    main()
