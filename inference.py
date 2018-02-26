#!/usr/bin/env python
# encoding: utf-8
import BatchDatsetReader as dataset

import tensorflow as tf
import numpy as np

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
MAX_CAPTCHA = 5
CHAR_SET_LEN = 10
batch_size = 64

train_dataset_reader = dataset.BatchDatset()
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

	#w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
	#w_c2_alpha = np.sqrt(2.0/(3*3*32))
	#w_c3_alpha = np.sqrt(2.0/(3*3*64))
	#w_d1_alpha = np.sqrt(2.0/(8*32*64))
	#out_alpha = np.sqrt(2.0/1024)

	# 3 conv layer
	w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
	b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
	conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv1 = tf.nn.dropout(conv1, keep_prob)

	w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
	b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
	conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv2 = tf.nn.dropout(conv2, keep_prob)

	w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 128]))
	b_c3 = tf.Variable(b_alpha*tf.random_normal([128]))
	conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
	conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv3 = tf.nn.dropout(conv3, keep_prob)

	w_c4 = tf.Variable(w_alpha*tf.random_normal([3, 3, 128,256]))
	b_c4 = tf.Variable(b_alpha*tf.random_normal([256]))
	conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
	conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv4 = tf.nn.dropout(conv4, keep_prob)
	# Fully connected layer
	w_d = tf.Variable(w_alpha*tf.random_normal([4*16*256, 1024]))
	b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
	dense = tf.reshape(conv4, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
	b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
	out = tf.add(tf.matmul(dense, w_out), b_out)
	#out = tf.nn.softmax(out)
	return out
def convert2gray(img):
	if len(img.shape) > 2:
		gray = np.mean(img, -1)

		# r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
		# gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return gray
	else:
		return img

def text2vec(text):
    vector = np.zeros(50)
    for i, c in enumerate(text):
        idx = i * 10 + int(c)
        vector[idx] = 1
    return vector

# text = '12345'
# a = text2vec(text)

def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % 10
        char_code = char_idx+ord('0')
        text.append(chr(char_code))
    return "".join(text)

def crack_captcha(sess,captcha_image,predict):


	text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

	text = text_list[0].tolist()
	vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
	i = 0
	for n in text:
		vector[i*CHAR_SET_LEN + n] = 1
		i += 1
	return vec2text(vector)

output = crack_captcha_cnn()
saver = tf.train.Saver()
predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
a = 0.0
with tf.Session() as sess:
	saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
	for itr in range(5000):
		train_images, train_annotations = train_dataset_reader.get_val_batch(itr,1)

		image = train_images[0]
		text = train_annotations[0]
		image = convert2gray(image)

		image = image.flatten() / 255

		predict_text = crack_captcha(sess,image,predict)
		if text == predict_text:
			a+=1.0
		print("正确: {}  预测: {}".format(text, predict_text))
print(a/5000)
