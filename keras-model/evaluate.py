from keras.models import load_model
import os
from captcha import Captcha, evaluate, Dataset
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model = load_model('model.h5.bak')
dataset = Dataset()
gen = dataset.gen(100)

nums = 0
acc = 0.0
for i in range(nums):
    X_test, y_test = next(gen)
    y_pred = model.predict(X_test)
    y_t = np.argmax(y_test, axis=2).T
    y_p = np.argmax(y_pred, axis=2).T
    for j in range(10):
        print(y_t[j], y_p[j])
    acc += evaluate(y_test, y_pred)
# print(acc / nums)
X_test, y_test = next(gen)
print(y_test[0].reshape((5, 10)))
