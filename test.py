import BatchDatsetReader as dataset
import os
import time
from model import Captcha

# load model
model = Captcha()
model.load_checkpoint('crack_capcha0.994400002956.model-9960')

# load dataset
dataset = dataset.BatchDatset(ratio=0.01, test_size=1000)
images, labels = dataset.get_val_batch(0, 1000)

count = len(labels)
correct = 0

# start timing
start = time.time()
t = start

# start testing
for i in range(count):
    image = images[i]
    label = labels[i]
    pred = model.predict(image)
    text = ''.join(map(str, pred))
    if text == label:
        correct += 1
    else:
        print(label, text)
        pass
    if i % 50 == 0:
        t = time.time() - t
        print('[%d/%d] average: %f'%(i, count, t / 50))
        t = time.time()

end = time.time()

# output result
print('Total time: %f s'%(end - start))
print('Average time: %f s'%((end - start) / count))
print('Total: %d, Correct: %d, Acc: %f'%(count, correct, float(correct) / float(count)))
