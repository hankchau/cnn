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
    x, y = data.load_training_data(roi_dir, save=True)

    # build CNN
    # model = CNN()
    # model.visualize()

    # Train


if __name__ == '__main__':
    main()
