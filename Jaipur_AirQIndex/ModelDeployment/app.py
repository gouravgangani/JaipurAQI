import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Data/Pickle_Models/xgboostRegressor.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods = ['Post'])
def predict():
    '''
    For rendering results from HTML File
    '''

    int_features = [int(x) for x in request.form.values]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    return render_template('index.html', prediction_text = 'Jaipur Air Quality Index: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)