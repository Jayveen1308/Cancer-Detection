from flask import Flask, render_template,request
import numpy as np
import pickle

app = Flask(__name__)

# load the ml model
model = pickle.load(open('xgb_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'f{i}']) for i in range(30)]
        prediction = model.predict([features])[0]
        result = 'Maligant' if prediction == 1 else 'Benign'
        return render_template('result.html',result=result)
    except:
        return render_template('result.html',result="Error:Invalid Input")

if __name__ == '__main__':
    app.run(debug=True)