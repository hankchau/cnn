from cnn import CNN
from data import get_data



def main():
    host = 'http://crs.comm.yzu.edu.tw:8888'
    get_data(host)

    # build CNN
    model = CNN()
    model.visualize()


if __name__ == '__main__':
    main()
