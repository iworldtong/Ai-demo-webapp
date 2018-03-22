import sys
import os
import glob
import re
import numpy as np

# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

import config.config as cfg




app = Flask(__name__)

# MODEL_PATH = './models/model1.h5'

# # Load trained model
# # model = load_model(MODEL_PATH)
# # model._make_predict_function()          # Necessary
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')


print('Model loaded. Start serving...')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/gen-anime', methods=['GET', 'POST'])
def upload():
    print(666)
    if request.method == 'POST':
        if True:#request.form['net'] == 'Anime Profile Generator':
            result = '1'
            print(result)
            return result
        else:
            result = '2'
            print(result)
            return result
    result = '3'
    print(result)
    return result



if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    http_server = WSGIServer(('', cfg.WEB_PORT), app)
    http_server.serve_forever()
