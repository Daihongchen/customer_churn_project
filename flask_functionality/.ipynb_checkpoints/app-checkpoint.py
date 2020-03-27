<<<<<<< HEAD
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('../src/model.pickle', 'rb'))
=======
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))
>>>>>>> 53b277c36dab15f3ffa062ab9554f5ab59e65070

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [x for x in request.form.values()]
    feature_names = [x for x in request.form.keys()]

    df = pd.DataFrame([features], columns = feature_names)

    df['international plan'] = (df['international plan'] == 'yes').astype(int)
    df['voice mail plan'] = (df['voice mail plan'] == 'yes').astype(int)

    df['total charge'] = df['total day charge'] + df['total eve charge'] + df['total intl charge'] + df['total night charge']
    df = df.drop(['total day charge', 'total eve charge', 'total intl charge', 'total night charge'], axis = 1)

    df = df.apply(pd.to_numeric)

    prediction = model.predict(df)[0]

    if prediction:
        return render_template('index.html', prediction_text = 'This customer is likely to leave soon.')
    else:
        return render_template('index.html', prediction_text = 'This customer will likely stay.')

<<<<<<< HEAD
=======
@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

>>>>>>> 53b277c36dab15f3ffa062ab9554f5ab59e65070
if __name__ == "__main__":
    app.run(debug=True)
