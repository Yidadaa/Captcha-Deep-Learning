#! python2
# coding: utf-8

"""
@file 用于下载数据集
@author Yidadaa
"""

import re
import os
import json
import io
from PIL import Image
import sys

if sys.version_info.major == 3:
    from urllib import request
elif sys.version_info.major == 2:
    import urllib2 as request

def getURLs(filename='./data-index/captcha_urls.csv'):
    """
    获取数据集的URL集合

    Args:
        @filename: URL集合文件的路径
    
    Return: URL字符串集合
    """
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

def downloadAndSave(urls, index_name='index'):
    """
    下载并保存爬取的数据
    
    Args:
        @urls(list): url集合
        @index_name(string): data目录下json索引文件的名称
            比如要想保存到data.json中，就传入'data'，默认是'index'
    """
    alreadyDownload = os.listdir('./data')
    total = len(urls)
    count = 0 # 用于统计已处理图片
    index = [] # 用于存放数据索引

    try:
        for url in urls:
            count += 1
            label = re.match(r'.*\/(\d+)\*.*', url).groups()[0] # 提取标签
            name = re.match(r'.*\*(\S+)\.gif', url).groups()[0] # 提取文件名

            filename = name + '.png'
            path =  './data/' + filename

            # 断点续存
            if filename not in alreadyDownload:
                try:
                    imgData = request.urlopen(url, timeout=3).read()
                except Exception:
                    continue # 添加超时处理
                gif = Image.open(io.BytesIO(imgData)) # 使用pil处理gif数据
                gif.save(path)

            index.append([path, label]) # 存放以[path, label]的形式的文件信息

            print(count, total)
    finally:
        with open('./data/{}.json'.format(index_name), 'w') as f:
            f.write(json.dumps(index)) # 将索引数据保存到{index_name}.json

        print('Index file saved at ./data/{}.json'.format(index_name))

if __name__ == '__main__':
    # 首先会从csv文件中读取url集合
    urls = getURLs('./data-index/captcha_test.csv')
    # 然后判断当前目录是否有data目录，若没有，则先创建data目录
    if 'data' not in os.listdir('./'):
        os.mkdir('./data')
    # 下载数据，并将图片以及标签信息保存到{index_name}.json中
    downloadAndSave(urls, index_name='test')
