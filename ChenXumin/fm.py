import numpy as np
import time
import sys

sys.path.append("..")
import preprocessing.preprocessing
np.seterr(divide='ignore',invalid='ignore')

def load_data(filename):
    data = open(filename)
    feature = []
    label = []
    for line in data.readlines():
        feature_tmp = []
        lines = preprocessing.outlier_detect(line.strip().split(','))
        if len(lines) == 0:
            continue
        for x in range(len(lines) - 2):
            feature_tmp.append(float(lines[x]))
        label.append(float(lines[-2])*2)
        feature.append(feature_tmp)
    data.close()
    return feature, label


def constructDataSet(feature):
    return np.mat(feature)


# loss function - MSE
def loss_function(label, y_hat):
    mse = np.square(label - y_hat).mean()
    return mse


def initialize_w_v(n, k):
    w = np.ones((n, 1))
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            v[i, j] = np.random.normal(0, 0.2)
    return w, v


def SGD(dataMatrix, classLabels, k, max_iter, learning_rate):
    m, n = np.shape(dataMatrix)
    w0 = 0
    w, v = initialize_w_v(n, k)
    print("start iteration")
    for it in range(max_iter):
        for x in range(m):
            v_1 = dataMatrix[x] * v  # x*v
            v_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)  # v^2*x^2
            interaction = 0.5 * np.sum(np.multiply(v_1, v_1) - v_2)

            p = w0 + dataMatrix[x] * w + interaction
            loss = loss_function(classLabels[x], p)
            w0 = w0 - learning_rate * loss * classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - learning_rate * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] - learning_rate * loss * classLabels[x] * (
                                    dataMatrix[x, i] * v_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

    return w0, w, v


def getAccuracy(w0, w, v):
    feature, labels = load_data("./ratings.csv")
    dataMatrix = np.mat(feature)
    m, n = np.shape(dataMatrix)
    allItem = 0
    error = 0
    result = []

    for x in range(m):
        allItem += 1
        v_1 = dataMatrix[x] * v  # x*v
        v_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)  # v^2*x^2
        interaction = 0.5 * np.sum(np.multiply(v_1, v_1) - v_2)

        p = w0 + dataMatrix[x] * w + interaction
        loss = loss_function(labels[x], p)
        result.append(loss)

        if loss == 0:
            continue
        else:
            error += 1
    print(result)
    return float(error)/allItem



start_training_time = time.time()
feature, label = load_data("./ratings_small.csv")
matrix = np.mat(feature)
w0, w, v = SGD(matrix, label, 10, 200, 0.3)
stop_training_time = time.time()
print("training time is %s"%(stop_training_time-start_training_time))
# print("\nAccuracy is %f"%(1-getAccuracy(w0,w,v)))
