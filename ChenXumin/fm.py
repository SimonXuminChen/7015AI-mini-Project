import numpy as np
import time

def load_data(filename):
    data=open(filename)
    feature=[]
    label=[]
    for line in data.readlines():
        feature_tmp=[]
        lines = line.strip().split('\t')
        for x in range(len(lines)-2):
            feature_tmp.append(float(lines[x]))
        label.append(int(lines[-2])*2-1)
        feature.append(feature_tmp)
    data.close()
    return feature,label

def initialize_w_v(n,k):
    w=np.ones(n,1)
    v = np.mat(np.zeros((n,k)))
    for i in range(n):
        for j in range(k):
            v[i,j] = np.random.normal(0,0.2)
    return w,v