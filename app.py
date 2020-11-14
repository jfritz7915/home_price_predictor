from flask import Flask, request
import numpy as np
import flasgger
from flasgger import Swagger
from pickle import load

app = Flask(__name__)
Swagger(app)


loaded_model = load(open('best_model.pk1','rb'))
poly = load(open('poly.pk1','rb'))
sc = load(open('scalar.pk1', 'rb'))

@app.route('/', methods=["Get"])
def predict():
    
    """House Price Prediction
    This is using docstrings for specifications.
    ---
    parameters:
        - name: House_Age
          in: query
          type: number
          description: "Enter Home age"
          required: true
        - name: Distance_to_the_nearest_MRT_station
          in: query
          type: number
          description: ""
          required: true
        - name: number_of_convienence_stores
          in: query
          type: number
          description: ""
          required: true
        - name: Latitude
          in: query
          type: number
          description: ""
          required: true
        - name: Longitude
          in: query
          type: number
          description: ""
          required: true
    responses:
          200:
              description: ""
    """
    
    l=[]
    i1=request.args.get('House_Age')
    l.append(i1)
    i2= request.args.get('Distance_to_the_nearest_MRT_station')
    l.append(i2)
    i3= request.args.get('number_of_convienence_stores')
    l.append(i3)
    i4=request.args.get('Latitude')
    l.append(i4)
    i5=request.args.get('Longitude')
    l.append(i5)
    
    arr= np.array([l])
    
    arr=poly.transform(arr)
    
    scaled_arr= sc.transform(arr)
    
    p=round(loaded_model.predict(scaled_arr)[0][0],2)
    
    return "Price of the house per unit area: " + str(p)
    
if __name__ =='__main__':
    app.run()
    