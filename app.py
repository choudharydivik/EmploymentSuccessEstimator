from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

application=Flask(__name__)

app=application

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictemployment',methods=['GET','POST'])
def predict_employment():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData