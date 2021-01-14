import data
# import cnn


def main():
    data.fill_data_index('data/areaMask.csv')
    data.tests.test_crop_image()
    data.tests.test_stack_training_data()



if __name__ == '__main__':
    main()
