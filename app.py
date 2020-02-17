import flask
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy
#import seaborn as sns

app = Flask(__name__, template_folder='templates')
pipe = pickle.load(open("pipe.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    data = pd.DataFrame({
        'fulltime_parttime': [args.get('fulltime_parttime')],
        'age_group' : [args.get('age_group')],
        'gender' : [(args.get('gender'))],
        'occupation': [args.get('occupation')]
        })

    employed_value = pipe.predict(data)

    employed_value = employed_value * 1000

    employed_value = numpy.round(employed_value)

    return render_template('result.html', employed_value = employed_value)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
