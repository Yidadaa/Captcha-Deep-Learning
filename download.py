"""
@file 用于下载数据集
@author Yidadaa
"""

import re
import os
import json
import io
from PIL import Image
from urllib import request

"""
获取数据集的URL集合
@filename: URL集合文件
@return: URL字符串集合
"""
def getURLs(filename = './captcha_urls.csv'):
    content = []
    urls = []
    reg = r'^http.*\.gif'

    with open(filename) as f:
        content = f.readlines()[1:]

    for l in content:
        p = re.match(reg, l)
        if p:
            urls.append(p.group())

    return urls

"""
下载并保存爬取的数据
@urls: list
@return: None
"""
def downloadAndSave(urls):
    alreadyDownload = os.listdir('./data')
    total = len(urls)
    count = 0 # 用于统计已处理图片
    index = [] # 用于存放数据索引

    for url in urls:
        count += 1
        label = re.match(r'.*\/(\d+)\*.*', url).groups()[0] # 提取标签
        name = re.match(r'.*\*(\S+)\.gif', url).groups()[0] # 提取文件名

        filename = name + '.png'
        path =  './data/' + filename

        # 断点续存
        if filename not in alreadyDownload:
            imgData = request.urlopen(url).read()
            gif = Image.open(io.BytesIO(imgData)) # 使用pil处理gif数据
            gif.save(path)

        index.append([path, label]) # index.json 存放以[path, label]的形式存放了文件信息

        print('已处理/总数： %d/%d'%(count, total), end='\r')

    with open('./data/index.json', 'w') as f:
        f.write(json.dumps(index)) # 将索引数据保存到index.json

if __name__ == '__main__':
    urls = getURLs()
    if 'data' not in os.listdir('./'):
        os.mkdir('./data')
    downloadAndSave(urls[0:5])