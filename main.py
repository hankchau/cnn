#from cnn import CNN
import data


def main():
    # scrape data from server
    host = 'http://crs.comm.yzu.edu.tw:8888'
    # data_path = 'csv_data/'
    # get_data(host, data_path)

    # data preprocessing
    data.fill_data_index('data/areaMask.csv')
    # data.find_label_distr()
    # extract ROI
    data.generate_heatmaps()

    # build CNN
    # model = CNN()
    # model.visualize()


if __name__ == '__main__':
    main()
