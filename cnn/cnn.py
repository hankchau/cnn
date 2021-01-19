import tensorflow as tf
import os


dropout1 = 0.5   # dropout rate 1
dropout2 = 0.5   # dropout rate 2
dropout3 = 0.5   # dropout rate 3


class CNN:
    def __init__(self):
        print('building CNN model ...')

        input = tf.keras.Input((256,86,3), batch_size=512, name='Input')

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
        fc1 = tf.keras.layers.Dense(32, name='FC1')(flat)
        drop3 = tf.keras.layers.Dropout(dropout3, name='Drop3')(fc1)

        output = tf.keras.layers.Dense(32, activation='softmax', name='Output')(drop3)

        # build model
        self.model = tf.keras.Model(inputs=input, outputs=output)

    def compile(self):
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['acc']
        )

    def visualize(self):
        print(self.model.summary())
        tf.keras.utils.plot_model(self.model, 'CNN.png')

    def fit(self):
        return 0

    def train_on_batch(self, x, y, class_weight=None, return_dict=False):
        dict = self.model.train_on_batch(
            x, y, class_weight=class_weight, return_dict=return_dict
        )
        print('Loss: %f     Accuracy: %f' % (dict['loss'], dict['acc']))
        print('Recall: ')
        return dict

    def save(self, outpath):
        outpath = os.path.join(outpath, 'model')
        self.model.save(outpath, overwrite=True)

    def save_weights(self, outpath):
        outpath = os.path.join(outpath, 'model_weights')
        self.model.save_weights(outpath, overwrite=True)
