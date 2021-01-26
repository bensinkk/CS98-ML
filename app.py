import os
import pickle
import pandas as pd
import sys
import numpy
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify
# # from flask_cors import CORS, cross_origin
import json

# app.py
app = Flask(__name__)


@app.route('/hello/', methods=['POST'])
def hello():
    jsdata = request.get_json()
    print(jsdata)

    y = []

    y = []
    for item in jsdata:
        y.append(int(jsdata[item]))

    array = numpy.array(y)
    array = numpy.reshape(array, (1, 7))

    loaded_model = pickle.load(open('saved_lin_reg.sav', 'rb'))
    result = loaded_model.predict(array)

    return jsonify(result.tolist())

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)