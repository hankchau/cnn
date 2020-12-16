import tensorflow as tf


dropout1 = 0.5   # dropout rate 1
dropout2 = 0.5   # dropout rate 2
dropout3 = 0.5   # dropout rate 3


class CNN_paper:
    def __init__(self):
        print('building CNN model ...')

        input = tf.keras.Input((40,40,256), batch_size=512, name='Input')

        # convolutional layers
        conv1 = tf.keras.layers.Conv2D(16, (3,3), padding='VALID', activation='relu', name='Conv1')(input)
        pool1 = tf.keras.layers.MaxPooling2D((2,2), padding='SAME', name='Pool1')(conv1)
        drop1 = tf.keras.layers.Dropout(dropout1, name='Drop1')(pool1)

        conv2 = tf.keras.layers.Conv2D(16, (3,3), padding='VALID', activation='relu', name='Conv2')(drop1)
        pool2 = tf.keras.layers.MaxPooling2D((2,2), padding='SAME', name='Pool2')(conv2)
        drop2 = tf.keras.layers.Dropout(dropout2, name='Drop2')(pool2)

        conv3 = tf.keras.layers.Conv2D(32, (3,3), padding='VALID', activation='relu', name='Conv3')(drop2)
        pool3 = tf.keras.layers.MaxPooling2D((2,2), padding='SAME', name='Pool3')(conv3)

        conv4 = tf.keras.layers.Conv2D(64, (5,5), padding='SAME', activation='relu', name='Conv4')(pool3)

        # fully-connected layers
        flat = tf.keras.layers.Flatten(name='Flat')(conv4)
        fc1 = tf.keras.layers.Dense(20, name='FC1')(flat)
        drop3 = tf.keras.layers.Dropout(dropout3, name='Drop3')(fc1)

        output = tf.keras.layers.Dense(1, activation='softmax', name='Output')(drop3)

        # build model
        self.model = tf.keras.Model(inputs=input, outputs=output)

    def visualize(self):
        print(self.model.summary())
        tf.keras.utils.plot_model(self.model, 'CNN.png')


class CNN_all():
    def __init__(self):
        input = tf.keras.Input()



        output = tf.keras.layers.Dense(6, activation='softmax', name='Output')()
        self.model = tf.keras.Model(inputs=input, outputs=output)

    def visualize(self):
        print(self.model.summary())


class FullyConnected():
    def __init__(self):
        input = tf.keras.Input()


        output = tf.keras.layers.Dense(6, activation='softmax', name='Output')
        self.model = tf.keras.Model(inputs=input, outputs=output)

    def visualize(self):
        print(self.model.summary())
