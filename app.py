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
        'occupation': [args.get('occupation')],
        'gender' : [args.get('gender')],
        'fulltime_parttime': [args.get('fulltime_parttime')],
        'age_group' : [args.get('age_group')]
        })

    employed_value = int(pipe.predict(data))


    #employed_value = numpy.round(employed_value) * 1000

    return render_template('result.html',  employed_value = employed_value)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
