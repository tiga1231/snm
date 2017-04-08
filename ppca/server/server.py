from flask import Flask
from flask import url_for, redirect
from flask import request
import json
from random import random
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.decomposition import PCA
from numpy.linalg import svd, inv, pinv
import sys

def noise(level):
    return np.random.randn() * level

app = Flask(__name__)

#generate data
phi = np.arange(0,2*np.pi, 2*np.pi / 12)
x = [[1.01*np.cos(i)+noise(.1), np.sin(i)+noise(.1), -.3] for i in phi]
x += [[1.01*np.cos(i)+noise(.1), np.sin(i)+noise(.1), .3] for i in phi]
x = np.array(x)


u,s,vt = svd(x)
sigmaPc = np.diag(s[:2]) #diag matrix w/ first 2 lambda
sigmaML2 = np.sum(s[2:]) / (len(s)-2) #estimation of epsilon noise

nullVect = np.array((vt.T)[:,2:].T)[0]

wML = (vt.T)[:,:2].dot( (sigmaPc - sigmaML2 * np.eye(2)) ** 0.5 )
sigmaR = wML.T.dot(wML) + sigmaML2 * np.eye(2)


def getZ(w, sigmaR):
    z = inv(  sigmaR  ).dot(w.T).dot(x.T)
    z = z.T
    return z

z = getZ(wML, sigmaR)



def getData(z):
    data = [{'i':i, 'x':d[0], 'y':d[1], 'tag':i} for i,d in enumerate(z)]
    return data

data = getData(z)



def pca(sd):
    u,l,vt = svd(sd)
    l = np.diag(l[:2])
    w = u[:,:2]
    return w


zUser = []
def update(d):
    global zUser,z, nullVect,x
    ziUser = np.array([d['i'], d['x'], d['y']])
    zUser.append(ziUser)
    
    if len(zUser) == 2:
        i,j = [int(zz[0]) for zz in zUser[:2]]
        
        dz0 = np.linalg.norm(z[i] - z[j])
        dz1 = np.linalg.norm(zUser[0][1:] - zUser[1][1:])

        dNull0 = np.linalg.norm((x[i] - x[j]).dot(nullVect))
        dNull1 = dNull0 * (dz1/dz0)
        delta = abs(dNull1 - dNull0)/2

        p11 = x[i] + delta * nullVect
        p12 = x[i] - delta * nullVect
        
        p21 = x[j] + delta * nullVect
        p22 = x[j] - delta * nullVect
        
        #moving away
        if dz1 > dz0:
            print 'away'
            if np.linalg.norm(p11 - p22) > dNull0:
                pU = np.array([p11, p22])
            else:
                pU = np.array([p12, p21])
        #moving closer
        else:
            print 'closer'
            if np.linalg.norm(p11 - p22) < dNull0:
                pU = np.array([p11, p22])
            else:
                pU = np.array([p12, p21])
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(pU[:,0], pU[:,1],pU[:,2])
        ax.scatter(x[:,0], x[:,1], x[:,2])
        plt.show()'''
        fp = pU.T.dot(pU)
        sd = x.T.dot(x)
        
        sd = 0.1*fp + sd * 0.9
        beta = pca(sd)
        z = x.dot(beta)
        dataNew = getData(z)
        zUser = []
        return dataNew
    else:
        global data
        return data


@app.route('/')
def index():
    global zUser
    zUser = []
    return redirect(url_for('static', filename='ppca.html'))


@app.route('/data', methods=['GET', 'POST'])
def dataReq():
    global data
    if request.method == 'POST':
        d = request.get_json()
        print '<POST> point', d['tag']
        data = update(d)
        #request.args.get('aa')
    return json.dumps(data)


if __name__ == '__main__':
    app.run()
