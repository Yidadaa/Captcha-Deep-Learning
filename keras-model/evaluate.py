from keras.models import load_model
import os
from captcha import Captcha

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model = load_model('model.h5')
X_test, y_test = next(Captcha().gen(1000))
y_pred = model.predict(X_test)


