from flask import Flask, redirect, url_for, request, render_template
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas as pd
import numpy as np
import pickle

# Define flask app
app = Flask(__name__)

labels = ['YES', 'NO']

input_data = [0]*len(labels)

# Load RFC model
best_model = pickle.load(open('models//heart.pkl', 'rb')) 

heart_df = pd.read_csv('dataset//heart_data.csv')
le = LabelEncoder()
heart_df['class_prognosis'] = le.fit_transform(heart_df['prognosis'])

print('\nModel loaded. Start serving...')
print('\nModel successfully loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input1 = request.form.get('Age')
        input2 = request.form.get('Sex')
        input3 = request.form.get('cp')
        input4 = request.form.get('trestbps')
        input5 = request.form.get('Cholestrol')
        input6 = request.form.get('fbs')
        input7 = request.form.get('ECG')
        input8 = request.form.get('thalach')
        input9 = request.form.get('exang')
        input10 = request.form.get('Oldpeak')
        input11 = request.form.get('Slope')
        input12 = request.form.get('CA')
        input13 = request.form.get('Thal')
        inputs = [input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13]
        for num in inputs:
            try:
                input_data[labels.index(num)] = 1
            except : pass

        rf_pred = best_model.predict(np.array([input_data]))[0]
        output_result = [rf_pred]
        # output_result.count(output_result)
        # dict( (l, output_result.count(l) ) for l in set(output_result))

        c = Counter(output_result)
        c = c.most_common(1)[0]
        result = le.inverse_transform([c[0]])[0]
        
        return render_template('index.html', col_list=labels, flag=True, input=inputs, input1=output_result, input2=dict( (l, output_result.count(l) ) for l in set(output_result)), input3=c, result=result )    

    return render_template('index.html', col_list=labels, flag=False)

if __name__ == '__main__':
    app.run(debug=True)

