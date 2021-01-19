import tensorflow as tf
import time


class ParkingLotDataset(tf.data.Dataset):
    def _generator(self, datapaths, num_samples):
        #Read data from memory
        for i in range(num_samples):

            datapaths[i]

    def __new__(cls, datapaths, num_samples=30000):
        return tf.data.Dataset.from_generator(
            cls._generator(datapaths, num_samples),
            output_types=tf.dtypes.float64,
            output_shapes=(),
            args=(num_samples, datapaths)
        )