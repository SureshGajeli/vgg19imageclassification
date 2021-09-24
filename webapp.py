# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:39:10 2021

@author: Shree
"""

import numpy as np
import os
#import re
#import blob

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model_path = 'vgg19.h5'

# load model
model = load_model(model_path)
model.make_predict_function()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    
    # preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method=="POST":
        # Get the file from the Post
        f = request.files['file']
        #save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        ### here we make our prediction
        pred = model_predict(file_path, model)
        pred_class = decode_predictions(pred, top=1)  # ImageNet Decode
        result = str(pred_class[0][0][1])              # Convert to String
        
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)