import tensorflow as tf
import data
import os
import sys
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import shuffle


dropout1 = 0.5   # dropout rate 1
dropout2 = 0.5   # dropout rate 2
dropout3 = 0.5   # dropout rate 3


class CNN:
    def __init__(self, outpath, model=None, weights=None):
        if model is None:
            self.model = self.build_model()
        else:
            self.model = tf.keras.models.load_model(model)
        if weights is not None:
            self.model.load_weights(weights)
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.outpath = outpath
        self.epochs = 0

    def build_model(self):
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
        return tf.keras.Model(inputs=input, outputs=output)

    def compile(self):
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['acc']
        )

    def visualize(self):
        print(self.model.summary())

    def fit(self, train, val, max_epochs=None, batch_size=512, early_stopping=0, save_model=False):
        outpath = os.path.join(self.outpath, 'cnn_model/')
        if os.path.isdir(outpath):
            shutil.rmtree(outpath)
        os.makedirs(outpath, exist_ok=True)
        os.mkdir(os.path.join(outpath, 'model_weights/'))

        if max_epochs is None:
            max_epochs = 300
        self.epochs = 0
        train_n = len(train)
        val_n = len(val)
        decreasing = 0
        tloss = []
        tacc = []
        vloss = []
        vacc = []
        stop = False

        # iterate through every epoch
        while not stop:
            sys.stderr.flush()
            print('Epoch: %i' % self.epochs)
            print('Training:')
            shuffle(train)
            for i in tqdm(range(0, train_n, batch_size)):
                batch_paths = train[i:i + batch_size]
                if i + batch_size >= train_n:
                    batch_paths = train[i:]
                x, y = data.load_batch(batch_paths, len(batch_paths))
                metrics = self.train_on_batch(x, y, return_dict=True)
                tloss.append(metrics['loss'])
                tacc.append(metrics['acc'])
                del x, y
            # print metrics
            avg_loss = sum(tloss) / len(tloss)
            avg_acc = sum(tacc) / len(tacc)
            self.train_loss.append(avg_loss)
            self.train_acc.append(avg_acc)
            print('\nTrain Loss: %f       Train Accuracy: %f' % (avg_loss, avg_acc))

            print('Validation:')
            # test on val for each epoch
            for i in tqdm(range(0, val_n, batch_size)):
                batch_paths = val[i:i + batch_size]
                if i + batch_size >= val_n:
                    batch_paths = val[i:]
                x, y = data.load_batch(batch_paths, len(batch_paths))
                metrics = self.model.test_on_batch(x, y, return_dict=True)
                vloss.append(metrics['loss'])
                vacc.append(metrics['acc'])
                del x, y
            # print metrics
            avg_loss = sum(vloss) / len(vloss)
            avg_acc = sum(vacc) / len(vacc)
            self.val_loss.append(avg_loss)
            self.val_acc.append(avg_acc)
            print('Validation Loss: %f       Validation Accuracy: %f' % (avg_loss, avg_acc))

            # check stopping criteria
            if self.epochs >= max_epochs - 1:
                stop = True
            if early_stopping > 0:
                if len(self.val_loss) >= early_stopping:
                    if self.val_loss[-1] > self.val_loss[-2]:
                        if decreasing >= early_stopping:
                            stop = True
                        else:
                            decreasing += 1
                    else:
                        decreasing = 0
            # save if model improves
            if self.epochs >= 1 and self.val_loss[-1] < self.val_loss[-2]:
                self.model.save_weights(os.path.join(outpath, 'model_weights/', 'weights_'+str(self.epochs)))
                if save_model:
                    self.model.save(os.path.join(outpath, 'model'))
            self.epochs += 1

    def train_on_batch(self, x, y, class_weight=None, return_dict=False):
        dict = self.model.train_on_batch(
            x, y, class_weight=class_weight, return_dict=return_dict
        )
        return dict

    def test_on_batch(self, x, y, return_dict=False):
        dict = self.model.test_on_batch(
            x, y, return_dict=return_dict
        )
        return dict

    def save(self, outpath):
        outpath = os.path.join(outpath, 'model')
        self.model.save(outpath, overwrite=True)

    def save_weights(self, outpath):
        outpath = os.path.join(outpath, 'model_weights')
        self.model.save_weights(outpath, overwrite=True)

    def plot_accuracy(self, outpath=None):
        if outpath is None:
            outpath = self.outpath
        outpath = os.path.join(outpath, 'cnn_accuracy.png')
        x = list(range(self.epochs))
        plt.plot(x, self.train_acc, '-b', label='train')
        plt.plot(x, self.val_acc, '-r', label='validation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('CNN Accuracy')
        plt.legend(loc='upper right')
        plt.savefig(outpath)
        plt.clf()
        plt.close()

    def plot_loss(self, outpath=None):
        if outpath is None:
            outpath = self.outpath
        outpath = os.path.join(outpath, 'cnn_loss.png')
        x = list(range(self.epochs))
        plt.plot(x, self.train_loss, '-b', label='train')
        plt.plot(x, self.val_loss, '-r', label='validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('CNN Categorical Cross-Entropy Loss')
        plt.legend(loc='upper right')
        plt.savefig(outpath)
        plt.clf()
        plt.close()
