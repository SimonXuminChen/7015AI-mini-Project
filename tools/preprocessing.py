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
    def __init__(self, filepath, train,is_print=False):
        self.feature_size = []
        self.feature=[]
        self.train = train

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

        if is_print:
            with open("./feature.txt","w") as f:
                for i, j,k in data[:, :3]:
                    user_feature = int(user_dict[i])
                    movie_feature = int(movie_dict[j])
                    content = str(user_feature)+","+str(movie_feature)+","+str(k)+"\n"
                    f.writelines(content)
                    self.feature.append([user_feature,movie_feature])
        else:
            for i, j,k in data[:, :3]:
                user_feature = int(user_dict[i])
                movie_feature = int(movie_dict[j])
                self.feature.append([user_feature,movie_feature])


    def __getitem__(self, index):
        """
        Xi is the features, also is the user_id and movie_id data, while Xv is de hidden vector
        """
        self.feature = np.array(self.feature)
        item = self.feature[index,:]
        target =self.label[index]
        Xi = torch.from_numpy(item.astype(np.int32)).unsqueeze(-1)
        Xv = torch.from_numpy(np.ones_like(item))
        # print(Xv)

        return Xi,Xv,target

    def __len__(self):
        return len(self.feature)



if __name__ == "__main__":
    test = PreProcessData("../data/ratings_small.csv", train=True)
    print(test.feature_size)
