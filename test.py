import BatchDatsetReader as dataset
import os
import time
from model import Captcha

model = Captcha()
#model.load_checkpoint('crack_capcha0.990800004005.model-9200')
model.load_checkpoint('crack_capcha0.994400002956.model-9960')

dataset = dataset.BatchDatset()
images, labels = dataset.get_val_batch(0, 5000)

correct = 0
start = time.time()
t = start
for i in range(len(labels)):
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
        print('[%d/%d] average: %f'%(i, len(labels), t / 50))
        t = time.time()

end = time.time()

print('Total time: %f s'%(end - start))
print('Average time: %f s'%((end - start) / len(labels)))
print('Total: %d, Correct: %d, Acc: %f'%(len(labels), correct, float(correct) / float(len(labels))))
