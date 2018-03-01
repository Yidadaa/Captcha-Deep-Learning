## 如何调用模型破解验证码
首先，安装项目所需的依赖项：
```
pip install -r requirements.txt
```

其次，从`CaptchaModel.py`中引入模型并初始化：
```python
from CaptchaModel import Captcha
model = Captcha()
model.load_checkpoint('crack_capcha0.994400002956.model-9960')
```

在执行`load_checkpoint`前，请保证本项目的根目录中存在`checkpoint`目录，并且里面包含如下形式的三个文件：
```
crack_capcha0.994400002956.model-9960.data-00000-of-00001
crack_capcha0.994400002956.model-9960.index
crack_capcha0.994400002956.model-9960.meta
```
如果没有，可以此获取训练好的模型：https://share.weiyun.com/dc274127a99be258aea284646b7faed1 下载后将里面的三个文件解压到`checkpoint`目录即可。

然后，对图片进行预处理，然后调用模型进行破解：
```python
import numpy as np
import scipy.misc as misc
import io

# 有两种方式，一种是从本地读取图片；另一种是从表单中获取图片数据
# 1. 从本地读取文件
f = open('path/to/image.gif').read()
# 2. 从表单中获取文件数据
f = request.files['files'].read()

# 读取完毕后，需要对图片进行放缩并转化为np.array数组
img = misc.imresize(misc.imread(io.BytesIO(f)), [64, 256], interp='nearest')
img_array = np.array(img)

# 调用模型进行破解
res = mode.predict(img_array)
# 识别结果res是一个长度为5的list
```
本项目包含了一个以Flask为例的文件，可以阅读[该文件](../server.py)的代码了解更多。

## 如何再次训练模型
假设您要训练自己的模型，训练数据保存在`data-index/my_data.csv`中，那么您可以按照以下流程重新训练模型。注意，如果您需要用gpu来加速训练，请安装gpu版的tensorflow，具体步骤请查阅tensorflow官方文档。

首先，使用`download.py`脚本下载训练数据，编辑该脚本的最后几行：
```python
if __name__ == '__main__':
    # 首先会从csv文件中读取url集合
    urls = getURLs('./data-index/my_data.csv')
    # 然后判断当前目录是否有data目录，若没有，则先创建data目录
    if 'data' not in os.listdir('./'):
        os.mkdir('./data')
    # 下载数据，并将图片以及标签信息保存到data/{index_name}.json中
    downloadAndSave(urls, index_name='my_data')
```
然后保存并运行`python download.py`开始下载数据。

数据下载完成后，使用`train.py`脚本来训练模型，编辑该脚本的第14行：
```python
# 修改index_file为上文提到'my_data'，不用加后缀名
train_dataset_reader = dataset.BatchReader(index_file='my_data')
```
然后保存并运行`python train.py`开始训练。

训练过程中，您看可以随时查看Test Accuracy，然后等待训练的结束。

训练完成后，模型将被自动保存在`checkpoint/`目录中，命名方式为：
```
crack_capcha0.{accuracy}.model-9960.data-00000-of-00001
crack_capcha0.{accuracy}.model-9960.index
crack_capcha0.{accuracy}.model-9960.meta
```
其中`accuracy`值代表了模型的准确度，越高越好。

然后，您就可以调用训练好的模型进行破解了。
