import pandas as pd
from flask import Flask, request, render_template
import pickle

import os, sys
module_path = os.path.abspath(os.pardir)
if module_path not in sys.path:
    sys.path.append(module_path)

from src.model_maker import convert_yes_no_to_numeric, combine_all_charges

app = Flask(__name__)
model = pickle.load(open('../src/model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [x for x in request.form.values()]
    feature_names = [x for x in request.form.keys()]

    df = pd.DataFrame([features], columns = feature_names)

    df = convert_yes_no_to_numeric(df)
    df = df.apply(pd.to_numeric)
    df = combine_all_charges(df)

    prediction = model.predict(df)[0]

    if prediction:
        return render_template('index.html', prediction_text = 'This customer is likely to leave soon.')
    else:
        return render_template('index.html', prediction_text = 'This customer will likely stay.')

if __name__ == "__main__":
    app.run(debug=True)
