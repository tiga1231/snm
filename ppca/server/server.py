from flask import Flask
from flask import url_for, redirect
from flask import request
import json
from random import random
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from numpy.linalg import svd, inv, pinv
import sys

def noise(level):
    return np.random.randn() * level

app = Flask(__name__)


def initData():
    #generate data
    phi = np.arange(0,2*np.pi, 2*np.pi / 12)
    x = [[1.01*np.cos(i)+noise(.05), np.sin(i)+noise(.05), -.2+noise(0.05)] for i in phi]
    x += [[1.01*np.cos(i)+noise(.05), np.sin(i)+noise(.05), .2+noise(0.05)] for i in phi]
    x = np.array(x)
    return x


def init():
    global x, data, zUser
    global a,b
    a,b = 0,1
    zUser = {}
    x = initData()
    u,s,vt = svd(x)
    #PCA
    w = (vt.T)[:,:2].dot(np.diag(s[:2]))
    z = x.dot(w)
    data = makeWebData(z)


def makeWebData(z):
    data = [{'i':i, 'x':d[0], 'y':d[1], 'tag':i, 'cat':0 if i<len(z)/2 else 1} 
                            for i,d in enumerate(z)]
    return data



def update(d):
    global zUser, x, z
    ziUser = np.array([ d['x'], d['y'] ])
    zUser[int(d['i'])] = ziUser
    print zUser
    if len(zUser) >= 2:
        zU = np.array(zUser.values())
        xU = x[zUser.keys()]
        
        global a,b
        lm = LinearRegression()
        lm.fit(np.concatenate([x,xU]), 
                np.concatenate([z,zU]), 
                np.concatenate( [a*np.ones(x.shape[0]), b*np.ones(xU.shape[0]) ] ))
        beta = lm.coef_.T
        '''fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(pU[:,0], pU[:,1],pU[:,2])
        ax.scatter(x[:,0], x[:,1], x[:,2])
        plt.show()'''
        zNew = x.dot(beta)
        dataNew = makeWebData(zNew)
        return dataNew
    else:
        global data
        return data


@app.route('/')
def index():
    global zUser
    zUser = {}
    return redirect(url_for('static', filename='ppca.html'))


@app.route('/resetdata', methods=['GET', 'POST'])
def reset():
    init()
    return redirect('/')

@app.route('/data', methods=['GET', 'POST'])
def dataReq():
    global data
    if request.method == 'POST':
        d = request.get_json()
        print '<POST> point', d['tag']
        data = update(d)
        #request.args.get('aa')
    return json.dumps(data)


@app.route('/setab', methods=['GET'])
def setab():
    global a,b
    a = request.args.get('a')
    b = request.args.get('b')
    a,b = float(a), float(b)
    print 'weights set to', a, b
    return 'getit'


a,b = 0,1
zUser = {}
x = initData()
u,s,vt = svd(x)
#PCA
w = (vt.T)[:,:2].dot(np.diag(s[:2]))
z = x.dot(w)
data = makeWebData(z)
if __name__ == '__main__':
    app.run()
