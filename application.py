from flask import Flask,redirect,render_template,jsonify,request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

## Import the models
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))
application=Flask(__name__)
app=application

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template('index.html')
    elif request.method=='POST':
        print(request.form)
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        
        new_data=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data)
        return render_template('index.html',result=round(result[0],4))
        

if __name__=="__main__":
    app.run(host="0.0.0.0")