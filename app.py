from flask import Flask, redirect, url_for, request, render_template, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Load RandomForestClassifier model
best_model = pickle.load(open('models//heart.pkl', 'rb')) 
labels = ['YES', 'NO']

heart_df = pd.read_csv('dataset//heart_data.csv')
le = LabelEncoder()
heart_df['class_prognosis'] = le.fit_transform(heart_df['target'])

print('\nModel loaded. Start serving...')
print('\nModel successfully loaded. Check http://127.0.0.1:5000/')

# Create all database tables
with app.app_context():
    db.create_all()

# Routes

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        session['username'] = username
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
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

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
