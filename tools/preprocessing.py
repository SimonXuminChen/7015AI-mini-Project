#  -*- coding: utf-8 -*-
'''
===================================================
@Time    : 2019/11/28  12:15 上午
@Author  : Simon Chen Xumin
@IDE     : PyCharm
===================================================
'''
import collections
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
import random


def load_data(filename):
    data = open(filename)
    feature = []
    label = []
    for line in data.readlines():
        feature_tmp = []
        lines = outlier_detect(line.strip().split(','))
        if len(lines) == 0:
            continue
        for x in range(len(lines) - 2):
            feature_tmp.append(float(lines[x]))
        label.append(float(lines[-2]) * 2)
        feature.append(feature_tmp)
    data.close()
    return feature, label


def outlier_detect(list, min=0., max=5.):
    flag = True
    new_list = []

    # check userid, movieid and rating is null
    for i in range(len(list) - 1):
        if list[i].isalpha():
            flag = False
            break
        elif (list[i] == "" or list[i] is None):
            # there is a null value,so we have to delete this line
            logging.warning("There is a null, so this might be processed")
            flag = False
            break
        else:
            new_list.append(list[i])

    if (not list[2].isalpha()) and ((float(list[2]) < min) or (float(list[2]) > max)):
        # there is a outlier value,so we have to delete this line
        logging.warning("There is outlier, so this might be processed")
        flag = False

    if flag is True:
        new_list.append(list[2])
        return new_list
    else:
        return []


class PreProcessData(Dataset):
    def __init__(self, filepath, train):
        self.feature_size = []
        self.feature=[]
        self.train = train

        if self.train:
            data = np.loadtxt(filepath, skiprows=1, delimiter=",")
            user_id = torch.from_numpy(data[:, 0])
            movie_id = torch.from_numpy(data[:, 1])
            self.label = torch.from_numpy(data[:, [2]])
            self.feature_size = [len(np.unique(user_id)), len(np.unique(movie_id))]
            user_dict = {}
            movie_dict = {}


            for i in range(0, self.feature_size[0]):
                user_dict[np.unique(user_id)[i]] = i

            for j in range(0, self.feature_size[1]):
                movie_dict[np.unique(movie_id)[j]] = j

            with open("./feature.txt","w") as f:
                for i, j,k in data[:, :3]:
                    user_feature = int(user_dict[i])
                    movie_feature = int(movie_dict[j])
                    content = str(user_feature)+","+str(movie_feature)+","+str(k)+"\n"
                    f.writelines(content)
                    self.feature.append([user_feature,movie_feature])


        else:
            data = np.loadtxt(filepath, skiprows=1, delimiter=",")
            user_id = data[:, 0]
            movie_id = data[:, 1]
            self.label = data[:, 2]
            self.feature_size = [len(np.unique(user_id)), len(np.unique(movie_id))]
            user_dict = {}
            movie_dict = {}

            for i in range(0, self.feature_size[0]):
                user_dict[np.unique(user_id)[i]] = i

            for j in range(0, self.feature_size[1]):
                movie_dict[np.unique(movie_id)[j]] = j

            with open("./feature.txt","w") as f:
                for i, j in data[:, :3]:
                    user_feature = int(user_dict[i])
                    movie_feature = int(movie_dict[j])
                    f.writelines(user_feature,movie_feature)
                    self.feature.append([user_feature,movie_feature])

    def __getitem__(self, index):
        self.feature = np.array(self.feature)
        item = self.feature[index,:]
        target =self.label[index]
        Xi = torch.from_numpy(item.astype(np.int32)).unsqueeze(-1)
        Xv = torch.from_numpy(np.ones_like(item))
        return Xi,Xv,target

    def __len__(self):
        return len(self.feature)

class DictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]



if __name__ == "__main__":
    test = PreProcessData("../data/ratings_small.csv", train=True)
    print(test.feature_size)
