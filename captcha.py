import BatchDatsetReader as dataset

import tensorflow as tf
import numpy as np

max_it =int(1e4 + 1)
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
MAX_CAPTCHA = 5
CHAR_SET_LEN = 10
batch_size = 64
file = open('acc.txt','a')

train_dataset_reader = dataset.BatchDatset()
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

# CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
	x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
	# 4 conv layer
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

	w_c4 = tf.Variable(w_alpha*tf.random_normal([3, 3, 128, 256]))
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
		#
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

output = crack_captcha_cnn()
#learning_rate = 0.001
global_step = tf.Variable(0,trainable = False)
lr = tf.train.exponential_decay(0.001, global_step, 4000*64, 0.1, staircase=True)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,global_step)

predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
max_idx_p = tf.argmax(predict, 2)
max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
correct_pred = tf.equal(max_idx_p, max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

acc_ = 0.0
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(max_it):
        train_images, train_annotations = train_dataset_reader.next_batch(batch_size)
        #print(train_images)
        batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
        batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])
        for k in range(batch_size):
            image = train_images[k]
            text = train_annotations[k]
            image = convert2gray(image)
            #print(image.shape)
            batch_x[k,:] = (image.reshape(-1)-128)/128 # (image.flatten()-128)/128  mean0
            batch_y[k,:] = text2vec(text)
            _, loss_,lr_ = sess.run([optimizer, loss,lr], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
        print(step, loss_,lr_)


        if step % 10 == 0:
            acc = 0.0
            bs = 100
            for itr in range(5000//bs):
                train_images, train_annotations = train_dataset_reader.get_val_batch(itr,bs)
    #print(train_images)
                batch_x_test = np.zeros([bs, IMAGE_HEIGHT*IMAGE_WIDTH])
                batch_y_test = np.zeros([bs, MAX_CAPTCHA*CHAR_SET_LEN])
                for k in range(bs):
                    image = train_images[k]
                    text = train_annotations[k]

                    image = convert2gray(image)
        #print(image.shape)
                    batch_x_test[k,:] = (image.reshape(-1)-128)/128 # (image.flatten()-128)/128  mean0
                    batch_y_test[k,:] = text2vec(text)
                acc_i,pre,l,p = sess.run([accuracy,predict,max_idx_l,max_idx_p], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
            #print(batch_y_test)
                acc += acc_i
            acc = acc/(5000//bs)
            print(step,'test_acc: ', acc)
            file.write(str(acc)+'\n')
            if acc > acc_:
                saver.save(sess, "./checkpoint/crack_capcha"+str(acc)+".model", global_step=step)
                acc_ = acc
