import numpy as np
import time
import sys
sys.path.append("..")
import preprocessing.preprocessing


def load_data(filename):
    data = open(filename)
    feature = []
    label = []
    for line in data.readlines():
        feature_tmp = []
        lines = preprocessing.outlier_detect(line.strip().split(','))
        if len(lines)==0:
            continue
        for x in range(len(lines) - 2):
            feature_tmp.append(float(lines[x]))
        label.append(lines[-2])
        feature.append(feature_tmp)
    data.close()
    return feature, label


def constructDataSet(feature):
    return np.mat(feature)


# loss function
def loss_function(label, y_hat):
    return 0.5 * np.linalg.norm(label, y_hat) ** 2


def initialize_w_v(n, k):
    w = np.ones(n, 1)
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            v[i, j] = np.random.normal(0, 0.2)
    return w, v


def SGD(dataMatrix, classLabels, k, max_iter, learning_rate):
    m, n = np.shape(dataMatrix)
    w0 = 0
    w, v = initialize_w_v(n, k)
    for it in range(max_iter):
        for x in range(m):
            v_1 = dataMatrix[x] * v
            v_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
            interaction = 0.5 * np.sum(np.multiply(v_1, v_1) - v_2)
            p = w0 + dataMatrix[x] * w + interaction
            # loss = loss_function(classLabels,prediction())
            w0 = w0 - learning_rate * loss * classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - learning_rate * loss * classLabels * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] * learning_rate * loss * classLabels[x] * (
                                    dataMatrix[x, i] * v_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

        if it % 1000 == 0:
            # print("iteartion: "+str(it)+", error: "+str(get_cost(prediction)))
            pass
    return w0, w, v


def prediction(x, w, v, w0):
    y = w0 + np.dot(w, x.T) + np.longlong(
        np.sum((np.dot(x, v) ** 2 - np.dot(x ** 2, v ** 2)), axis=1).reshape(len(x), 1)) / 2.0
    return y


start_time = time.time()

feature, label = load_data("./ratings_small.csv")
print(label)
end_time = time.time()
