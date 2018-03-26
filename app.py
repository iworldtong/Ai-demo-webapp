import sys
import os
import glob
import re
import numpy as np

from flask import Response, send_file
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

import config as cfg
from tools import *
from net import *



app = Flask(__name__)



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/gen-anime', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':     
        img_path = gen_anime_profile(save_dir=cfg.TEMP_DIR)        
        img_stream = return_img_stream(img_path, remove_later=True)        
        return img_stream        
    return None





if __name__ == '__main__':
    
    # app.run(port=5002, debug=True)
    
    print('Start serving...')

    http_server = WSGIServer(('', cfg.WEB_PORT), app)
    http_server.serve_forever()
