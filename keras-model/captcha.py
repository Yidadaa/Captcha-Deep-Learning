#!/usr/bin/env python
# encoding: utf-8

from keras.utils.np_utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from PIL import Image
import numpy as np
import json
import os

class Evaluate(Callback):
    """模型的评价函数类

    该类继承自keras.callbacks.Callback
    """
    def __init__(self, model, gen, batch_num=10):
        self.acc = []
        self.gen = gen
        self.model = model
        self.batch_num = batch_num

    def on_epoch_end(self, epoch, logs=None):
        acc = 0.0
        data_gen = self.gen(128)
        for i in range(self.batch_num):
            X_test, y_test = next(data_gen)
            y_pred = self.model.predict(X_test)

        self.accs.append(acc)
        print('\nCurrent Acc: %f\n'%acc)

class Captcha:

    def __init__(self):
        self.width, self.height, self.n_len, self.n_class = 240, 60, 5, 10
        self.model = None
        self.init_model()

    def gen(self, batch_size=96):
        width, height = self.width, self.height
        X = np.zeros((batch_size, height, width, 1), dtype=np.uint8)
        y = [np.zeros((batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        path_prefix = '../data/'
        file_path = path_prefix + 'index.json'
        with open(file_path) as f:
            src = json.loads(f.read())
            src = np.array(src)
        labels = src[:, 1]
        images = src[:, 0]

        all_batch = int(len(src) / batch_size) - 1
        this_batch = 0
        while True:
            for i in range(batch_size):
                index = this_batch * batch_size + i
                label_str = labels[index]
                X[i] = np.array(Image.open(path_prefix + images[index].replace('./data/', ''))).reshape(height, width, 1)
                for j, c in enumerate(label_str):
                    y[j][i, :] = 0
                    y[j][i, int(c)] = 1
            this_batch = (this_batch + 1) % all_batch

            # print(this_batch, all_batch)
            yield X, y

    def init_model(self):
        width, height = self.width, self.height
        input_tensor = Input((height, width, 1))
        x = input_tensor
        for i in range(4):
            x = Convolution2D(32*2**i, (2, 2), activation='relu')(x)
            x = Convolution2D(32*2**i, (2, 2), activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            print('adding %d'%i)

        x = Flatten()(x)
        x = Dropout(0.25)(x)
        x = [Dense(self.n_class, activation='softmax', name='c%d'%(i + 1))(x) for i in range(self.n_len)]
        self.model = Model(inputs=input_tensor, outputs=x)
        self.model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

    def train(self):
        self.model.fit_generator(self.gen(128), steps_per_epoch=1000, epochs=5,
            validation_data=self.gen(256), validation_steps=10)
        # 训练完成后，将模型保存起来
        self.model.save_weights(self.checkpoint_path)
        print('\nModel has been saved at %s'%self.checkpoint_path)

    def predict(self, X):
        self.model.predict(X)

    def loads(self, checkpoint_path='./model.h5'):
        if os.path.exists(self.checkpoint_path):
            # 自动从上次训练数据中恢复
            self.model.load_weights(self.checkpoint_path)
            print('\nModel has been loaded from %s'%self.checkpoint_path)
        else:
            print('\nNo model to laod!')

