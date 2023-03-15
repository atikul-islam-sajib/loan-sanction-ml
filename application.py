from flask import Flask, render_template, render_template_string, request
import cv2
import numpy as np
import pandas as pd
import urllib.request
import os
import keras
import pickle
import base64
import pickle

import tensorflow
import keras.utils as image
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from keras.utils import load_img,  img_to_array
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor

application = Flask(__name__)
app = application

@app.route("/")
def hello_world():
    return render_template('index.html')

UPLOAD_FOLDER = 'static'

application = Flask(__name__)
app = application

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit_file():
    list_all  = [] 
    amount    = float(request.form['amount'])
    list_all.append(amount)

    credit    = float(request.form['credit'])
    list_all.append(credit)

    applicant = float(request.form['applicant'])
    list_all.append(applicant)

    income    = str(request.form['income'])

    if income.upper() == 'HIGH':
        list_all.extend([1, 0, 0])
    elif income.upper() == 'LOW':
        list_all.extend([0, 1, 0])
    else:
        list_all.extend([0, 0, 1])

    property_  = float(request.form['property'])
    list_all.append(property_)

    print(list_all)

    with open('XGBoostRegressor.pkl', 'rb') as f:
        XGboostRegressor = pickle.load(f)

    prediction = XGboostRegressor.predict(np.expand_dims(np.array(list_all), axis = 0))


    return render_template('index.html', result = prediction[0])

if __name__=="__main__":
    application.run(host="0.0.0.0", port = 5005)
