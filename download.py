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
    index = 0
    info = []

    for url in urls:
        index += 1
        label = re.match(r'.*\/(\d+)\*.*', url).groups()[0] # 提取标签
        name = re.match(r'.*\*(\S+)\.gif', url).groups()[0] # 提取文件名

        path = './data/' + name
        sourceImg = '/'.join(['.', 'data', name, name]) + '.gif'

        # 断点续存
        if name not in alreadyDownload:
            os.mkdir(path)

            imgData = request.urlopen(url).read()

            with open(sourceImg, 'wb') as f:
                f.write(imgData)

            gif = Image.open(io.BytesIO(imgData)) # 使用pil处理gif数据
            i = 0
            palette = gif.getpalette()
            try:
                while True:
                    gif.putpalette(palette)
                    newImg = Image.new('RGBA', gif.size)
                    newImg.paste(gif)
                    newImg.save(path + '/' + str(i) + '.png')

                    i += 1
                    gif.seek(gif.tell() + 1)
            
            except EOFError:
                pass

        print('已处理/总数： %d/%d'%(index, total), end='\r')
        info.append({
            'name': name, # hash值
            'label': label, # 标签
            'path': path, # 图片路径
            'sourceImg': sourceImg, # 原始图片
            'frameFormat': 'png', # 帧图片格式
            'frameCount': 20 # 一共多少帧
        })

    with open('./data/info.json', 'w') as f:
        f.write(json.dumps(info)) # 将索引数据保存到info.json

if __name__ == '__main__':
    urls = getURLs()
    if 'data' not in os.listdir('./'):
        os.mkdir('./data')
    downloadAndSave(urls)