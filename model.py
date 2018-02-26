import tensorflow as tf
import numpy as np

class Captcha:
    def __init__(self):
        self.MAX_CAPTCHA = 5
        self.CHAR_SET_LEN = 10
        self.batch_size = 64
        self.WIDTH = 256
        self.HEIGHT = 64
        self.sess = tf.Session()
        self.X = tf.placeholder(tf.float32, [None, self.WIDTH * self.HEIGHT])
        self.keep_prob = tf.placeholder(tf.float32)
        self.output = self.cnn_graph()

    def cnn_graph(self, w_alpha=0.01, b_alpha=0.1):
        X = self.X
        # define compute graph of cnn
        keep_prob = self.keep_prob
        x = tf.reshape(X, shape=[-1, self.HEIGHT, self.WIDTH, 1])
        # convolutional layers
        for i in range(4):
            dim = [1, 32, 64, 128]
            w_c = tf.Variable(w_alpha*tf.random_normal([3, 3, dim[i], 32 * 2**i]))
            b_c = tf.Variable(b_alpha*tf.random_normal([32 * 2**i]))
            x = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c, strides=[1, 1, 1, 1], padding='SAME'), b_c))
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            x = tf.nn.dropout(x, keep_prob)
        # dense layer
        w_d = tf.Variable(w_alpha * tf.random_normal([4 * 16 * 256, 1024]))
        b_d = tf.Variable(w_alpha * tf.random_normal([1024]))
        dense = tf.reshape(x, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)
        # outpu layer
        w_out = tf.Variable(w_alpha*tf.random_normal([1024, self.MAX_CAPTCHA * self.CHAR_SET_LEN]))
        b_out = tf.Variable(b_alpha*tf.random_normal([self.MAX_CAPTCHA * self.CHAR_SET_LEN]))
        out = tf.add(tf.matmul(dense, w_out), b_out)

        return out


    def load_checkpoint(self):
        tf.train.Saver().restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))
        print('Model restored from checkpoint.')

    def predict(self, image):
        if len(image.shape) > 2:
            image = np.mean(image, -1)
        image = (image.flatten() - 128) / 128

        predict = tf.argmax(tf.reshape(self.output, [-1, self.MAX_CAPTCHA, self.CHAR_SET_LEN]), 2)
        text = self.sess.run(predict, feed_dict={ self.X: [image], self.keep_prob: 1 })
        text = text[0].tolist()
        return text
