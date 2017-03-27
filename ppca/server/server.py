from flask import Flask
from flask import url_for, redirect
from flask import request
import json
from random import random


app = Flask(__name__)

data = [{'i':i, 'x':i, 'y':i+random(), 'tag':i} for i in range(10)]


@app.route('/')
def index():
    return redirect(url_for('static', filename='ppca.html'))


@app.route('/data', methods=['GET', 'POST'])
def getData():
    global data
    if request.method == 'POST':
        d = request.get_json()
        print '<POST>', d
        d['x'] += random()
        d['y'] += random()
        data[d['i']] = d
        #request.args.get('aa')
    return json.dumps(data)


if __name__ == '__main__':
    app.run()
