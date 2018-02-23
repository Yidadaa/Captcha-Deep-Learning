#!/usr/bin/env python
# encoding: utf-8


from captcha import Captcha
import os
from keras.utils import plot_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 指定使用卡1

model = Captcha()
model.train()
# next(model.gen())
# plot_model(model.model, to_file="model.png", show_shapes=True)

