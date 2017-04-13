from flask import Flask
from flask import url_for, redirect
from flask import request
import json
from random import random
import numpy as np
from scipy.optimize import minimize


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



def obj(beta, x, z, w):
    '''objective function'''
    beta = beta.reshape([x.shape[1],2])
    w = np.diag(w)
    a = x.dot(beta)-z
    return a.T.dot(w).dot(a)

def der(beta, x, z, w):
    '''derivative'''
    beta = beta.reshape([x.shape[1],2])
    return 2*x.T.dot(np.diag(w)).dot(x.dot(beta)-z)


def c1(beta):
    '''constraint 1'''
    beta = beta.reshape([-1,2])
    res = beta[:,0].dot(beta[:,0]) - beta[:,1].dot(beta[:,1])
    return res
def jac1(beta):
    return np.array([2*beta[i] if i%2==0 
                    else -2*beta[i] 
                    for i in range(len(beta))])

def c2(beta):
    '''constraint 2'''
    beta = beta.reshape([-1,2])
    res = beta[:,1].dot(beta[:,0])
    return res
def jac2(beta):
    return np.array([beta[i+1] if i%2==0 else beta[i-1] for i in range(len(beta))])

def c3(beta):
    beta = beta.reshape([-1,2])
    res = beta[:,1].dot(beta[:,1]) - 1
    return res
def jac3(beta):
    return np.array([2*beta[i] if i%2==1 else 0 for i in range(len(beta))])

def append1(x):
    return np.concatenate([x, np.ones([x.shape[0],1])], axis=1)

def update(d):
    global zUser, x, z
    ziUser = np.array([ d['x'], d['y'] ])
    zUser[int(d['i'])] = ziUser
    if len(zUser) >= 2:
        zU = np.array(zUser.values())
        xU = x[zUser.keys()]
        
        global a,b
        cons = (
                {'type': 'eq', 'fun' : c1, 'jac' : jac1}, #bt b = I*k
                {'type': 'eq', 'fun' : c2, 'jac' : jac2},
                #{'type': 'eq', 'fun' : c3, 'jac' : jac3},
                )
        beta0 = np.random.random([x.shape[1]+1,z.shape[1]])
        
        if a == 0:
            weight = np.ones(zU.shape[0])
            xx = xU
            zz = zU

        elif b==0:
            weight = np.ones(z.shape[0])
            xx = x
            zz = z
            
        else :
            weight = np.concatenate([  a/z.shape[0]  *np.ones(z.shape[0]),
                                    b/zU.shape[0]  *np.ones(zU.shape[0])])
            xx = np.concatenate([x,xU])
            zz = np.concatenate([z,zU])
        
        xx = append1(xx)
        res = minimize(obj, beta0, args=(xx,zz,weight), jac=der,
                    constraints=cons, 
                    options={'disp': True, 
                            'maxiter': 1000,
                            'ftol':1e-5},
                    method='SLSQP',
                    )
        beta = res.x.reshape([-1,2])
        
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(xU[:,0], xU[:,1],xU[:,2])
        ax.scatter(x[:,0], x[:,1], x[:,2])
        plt.show()'''
    
        zNew = append1(x).dot(beta)
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
        print '<POST> point', d['tag'], '(%.2f, %.2f)' % (d['x'], d['y'])
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
