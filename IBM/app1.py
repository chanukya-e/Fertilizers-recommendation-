import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import requests

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from werkzeug.utils import secure_filename
from flask import Flask, redirect,render_template, request,url_for



app = Flask(__name__)
model=load_model("fruit.h5")
model1=load_model("vegetable.h5")

@app.route('/')
def home():
    return render_template('Home.html')
@app.route('/Prediction')
def prediction():
    return render_template('Predict.html')

@app.route('/Predict',methods=['POST'])
def predict():
    if request.method =='POST':
        f= request.files['images']
        basepath = os.path.dirname(__file__)
        file_path=os.path.join(basepath, 'uploads',secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        plant=request.form['plant']
        print (plant)
        if (plant=="vegetable"):
             preds = model1.predict (x)
             preds=np.argnax (preds)
             print (preds)
             df=pd.read_excel("precautions - veg.xlsx")
             print (df. iloc(preds) ['caution'])
        else:
            preds=model.predict(x)
            preds=np.argmax(preds)
            df=pd.read_excel("precautions - fruits.xlsx")
            print(df.iloc[preds]['caution'])
        return df.iloc[preds]['caution']
if __name__=="__main__":
     app.run(debug=False)


