#!/usr/bin/env python
# encoding: utf-8

from keras.utils.np_utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras import optimizers
from keras import backend as K
from PIL import Image
import numpy as np
import json
import os


def evaluate(y_true, y_pred):
    """评估函数
    Args:
        y_true(List): 正确标签
        y_pred(List): 预测结果
    Return:
        Acc(float): 准确率
    """
    # y_true_t = np.argmax(y_true.reshape(y_true.shape[0], 5, 10), axis=2).T
    # y_pred_t = np.argmax(y_pred.reshape(y_true.shape[0], 5, 10), axis=2).T
    y_true_t = np.argmax(y_true, axis=2).T
    y_pred_t = np.argmax(y_pred, axis=2).T
    acc = np.mean(map(np.array_equal, y_pred_t, y_true_t))
    print("\n\n\033[0;31mCurrent acc: %f%%\033[0m\n\n" % (acc * 100))
    return acc

class Evaluator(Callback):
    """模型的评价函数类

    该类继承自keras.callbacks.Callback
    """
    def __init__(self, dataset):
        """
        Args:
            dataset(Dataset): 数据集
        """
        self.accs = []
        self.gen = dataset.gen(type='test')

    def on_epoch_end(self, epoch, logs=None):
        X_test, y_test = next(self.gen)
        y_pred = self.model.predict(X_test)
        acc = evaluate(y_test, y_pred)
        self.accs.append(acc)

def sig_ce_loss(target, output):
    return K.categorical_crossentropy(target, output, from_logits=True)


class Dataset():
    """数据集读取类
    """
    def __init__(self, root='../data/', index_file='index.json', test_count=5000):
        self.path_prefix = root
        self.file_path = root + index_file
        self.test_count = test_count

        self.train_index = []
        self.train_label = []
        self.test_index = []
        self.test_label = []
        self.load_data()

    def load_data(self):
        """读取索引数据到内存
        """
        with open(self.file_path) as f:
            src = json.loads(f.read())
            src = np.array(src)
        labels = src[:, 1]
        images = src[:, 0]
        N = len(labels)
        self.train_index = images[0:N - self.test_count]
        self.train_label = labels[0:N - self.test_count]
        self.test_index = images[N - self.test_count:]
        self.test_label = labels[N - self.test_count:]
        print('\nData loading done! %d pcs train data and %d pcs test data loaded.\n'%(N, self.test_count))

    def gen(self, batch_size=128, type='train'):
        """读取图像数据并返回

        Args:
            batch_size(Int): 单个Batch包含样本数目
            type(String): 'train' or 'test' 用于复用数据生成器

        Return:
            X(Array): 训练数据
            y(List): 训练label
        """
        width, height = 256, 64
        n_class, n_len = 10, 5 # 需要识别10个字符，5位数

        images = self.train_index if type == 'train' else self.test_index
        labels = self.train_label if type == 'train' else self.test_label
        if type == 'test':
            batch_size = self.test_count # 一次性返回所有测试数据

        X = np.zeros((batch_size, height, width, 1), dtype=np.uint8)
        y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]

        path_prefix = self.path_prefix # 数据文件前缀
        all_batch = int(len(images) / batch_size) # 一个epoch包含的batch
        this_batch = 0 # 当前batch
        while True:
            for i in range(batch_size):
                index = this_batch * batch_size + i
                label_str = labels[index]
                X[i] = np.array(Image.open(path_prefix + images[index].replace('./data/', '')).resize((64, 256))).reshape(height, width, 1)
                for j, c in enumerate(label_str):
                    y[j][i, :] = 0
                    y[j][i, int(c)] = 1
            this_batch = (this_batch + 1) % all_batch

            # print(this_batch, all_batch)
            # yield X, np.array(y).reshape(batch_size, n_class * n_len)
            yield X, y

class Captcha:
    def __init__(self):
        self.width, self.height, self.n_len, self.n_class = 256, 64, 5, 10
        self.model = None
        self.dataset = Dataset()
        self.init_model()

    def init_model(self):
        width, height = self.width, self.height
        input_tensor = Input((height, width, 1))
        x = input_tensor
        for i in range(4):
            x = Convolution2D(32*2**i, (3, 3), activation='relu', padding='same')(x)
            # x = Convolution2D(32*2**i, (2, 2), activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            if i < 2:
                x = normalization.BatchNormalization(epsilon=1e-6)(x)

        x = Flatten()(x)
        # x = Dense(self.n_len * self.n_class, activation='softmax')(x)
        # x = Reshape((self.n_len, self.n_class))(x)
        x = [Dense(self.n_class, activation='softmax', name='c%d'%(i + 1))(x) for i in range(self.n_len)]
        self.model = Model(inputs=input_tensor, outputs=x)
        sgd = optimizers.Adam(0.001)
        self.model.compile(loss=sig_ce_loss,
                optimizer='adadelta',
                metrics=['accuracy'])

    def train(self):
        self.checkpoint_path = './model.h5'
        custom_acc = Evaluator(self.dataset)
        history = self.model.fit_generator(self.dataset.gen(128), steps_per_epoch=200, epochs=100,
            validation_data=self.dataset.gen(256), validation_steps=1,
            callbacks=[custom_acc])
        # save acc and model
        self.model.save(self.checkpoint_path)
        with open('./acc.json', 'w') as f:
            f.write(json.dumps(custom_acc.accs))
        print('History acc:', custom_acc.accs)
        print('\nModel has been saved at %s'%self.checkpoint_path)

    def predict(self, X):
        self.model.predict(X)
