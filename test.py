import BatchDatsetReader as dataset
import os
from model import Captcha

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model = Captcha()
model.load_checkpoint()

dataset = dataset.BatchDatset()
images, labels = dataset.get_val_batch(0, 100)
image = images[0]
label = labels[0]

for i in range(10):
    image = images[i]
    label = labels[i]
    pred = model.predict(image)
    print(label, pred)
