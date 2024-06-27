from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = model.predict(input_data_reshaped)
        
        if prediction[0] == 0:
            result = 'The Person has Heart Disease'
        else:
            result = 'The Person does not have a Heart Disease'
        
        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
