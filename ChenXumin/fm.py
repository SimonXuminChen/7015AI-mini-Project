import numpy as np
from ChenXumin.preprocessing import outlier_detect
import time

def load_data(filename):
    data=open(filename)
    feature=[]
    label=[]
    for line in data.readlines():
        feature_tmp=[]
        lines = line.strip().split(',')
        for x in range(len(lines)-2):
            flag = outlier_detect(lines[x])
            if not flag:
                break
            feature_tmp.append(float(lines[x]))
        label.append(int(lines[-2]))
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

def sigmoid(x):
    return 1/(1+np.exp(-x))