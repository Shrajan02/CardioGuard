from flask import Flask, redirect, url_for, request, render_template
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle

# Define flask app
app = Flask(__name__)

labels = ['YES', 'NO']

# Load RandomForestClassifier model
best_model = pickle.load(open('models//heart.pkl', 'rb')) 

heart_df = pd.read_csv('dataset//heart_data.csv')
le = LabelEncoder()
heart_df['class_prognosis'] = le.fit_transform(heart_df['target'])

print('\nModel loaded. Start serving...')
print('\nModel successfully loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        inputs = [
            request.form.get('Age'),
            request.form.get('Sex'),
            request.form.get('cp'),
            request.form.get('trestbps'),
            request.form.get('Cholestrol'),
            request.form.get('fbs'),
            request.form.get('restecg'),
            request.form.get('thalach'),
            request.form.get('exang'),
            request.form.get('Oldpeak'),
            request.form.get('slope'),
            request.form.get('CA'),
            request.form.get('thal')
        ]

        # Convert inputs to numeric values
        inputs = [float(val) if val else 0.0 for val in inputs]

        # Convert Sex, cp, fbs, restecg, exang, slope, and thal to integers
        for i in [1, 2, 5, 6, 8, 10, 12]:
            inputs[i] = int(inputs[i])

        # Predict using the model
        rf_pred = best_model.predict([inputs])[0]
        result = le.inverse_transform([rf_pred])[0]

        return render_template('index.html', flag=True, result=result)

    return render_template('index.html', flag=False)

if __name__ == '__main__':
    app.run(debug=True)
