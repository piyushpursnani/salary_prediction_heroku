import re
from flask import Flask,render_template, request
import joblib


app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def welcome():
    return render_template('base.html')

@app.route('/predict', methods = ['post'])
def predict():

    exp = request.form.get('experience')
    score = request.form.get('test_score')
    intervie_score = request.form.get('interview_score')

    prediction = model.predict([[int(exp),int(score), int(intervie_score)]])

    output = prediction

    return render_template('base.html', prediction_text = f"empluee {output}")


if __name__ == '__main__':
    app.run(debug=True)