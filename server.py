#!/usr/bin/env python
# encoding: utf-8

from flask import Flask, request, Response
import numpy as np
import scipy.misc as misc

from model import Captcha

import io
import json

from gevent import monkey
from gevent.pywsgi import WSGIServer
# support async
monkey.patch_all()

model = Captcha()
model.load_checkpoint('crack_capcha0.994400002956.model-9960')

app = Flask(__name__)

@app.route('/crack', methods=['GET', 'POST'])
def crack_upload_img():
    if request.method == 'POST':
        try:
            f = request.files['file']
            img = f.read()
            gif = misc.imresize(misc.imread(io.BytesIO(img)), [64, 256], interp='nearest')
            img_array = np.array(gif)
            text = model.predict(img_array)
            res = {}
            res['result'] = ''.join(map(str, text))
            return Response(json.dumps(res), mimetype='application/json')
        except:
            res = { "err": "Error! Please check your file." }
            return Response(json.dumps(res), mimetype='application/json')
    return '''
        <!doctype html>
        <title>Upload your file</title>
        <h1>Upload File Here</h1>
        <form action="" method=post enctype=multipart/form-data>
            <p><input type=file name=file>
            <input type=submit value=Upload>
        </form>
    '''
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=80)
    http_server = WSGIServer(('', 80), app)
    http_server.serve_forever()
