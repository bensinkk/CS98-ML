import os
import pickle
import pandas as pd
import sys
import numpy
from sklearn.linear_model import LinearRegression
from flask import Flask
from flask import request
from flask import jsonify
import json


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello', methods=['POST'])
    def hello():
        jsdata = request.get_json()
        print(jsdata)

        y = []

        y = []
        for item in jsdata:
            y.append(int(jsdata[item]))

        array = numpy.array(y)
        array = numpy.reshape(array, (1, 8))
    

        loaded_model = pickle.load(open('app/saved_lin_reg.sav', 'rb'))
        result = loaded_model.predict(array)
     
        return jsonify(result.tolist())

    return app
