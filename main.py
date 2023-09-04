import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy  as np


app=Flask(__name__)
data=pd.read_csv('cleaned_data.csv')
pipe=pickle.load(open("RidgeModel.pk1","rb"))

@app.route('/')

def index():
    locations=sorted(data['location'].unique())

    return render_template('prediction.html',locations=locations)
     


@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('total_sqft')
    print(location, bhk, bath, sqft)
    input=pd.DataFrame([[location,bhk,bath,sqft]],columns=['location','bhk','bath','total_sqft'])
    predict=pipe.predict(input)[0] * 1e5

    return str(np.round(predict,2))

if __name__=="__main__":
    app.run(debug=True,port=5001)
