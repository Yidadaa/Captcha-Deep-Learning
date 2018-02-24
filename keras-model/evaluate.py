from keras.models import load_model
import os
from captcha import Captcha
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model = load_model('model.h5')
dataset = Captcha()

nums = 20
acc = 0.0
for i in range(nums):
    X_test, y_test = next(dataset.gen(1000))
    y_pred = model.predict(X_test)
    pred = np.argmax(y_pred, axis=2).T
    y_true = np.argmax(y_pred, axis=2).T
    acc += np.mean(map(np.array_equal, pred, y_true))
print(acc / nums)
