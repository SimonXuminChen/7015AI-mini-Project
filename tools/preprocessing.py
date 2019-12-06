#  -*- coding: utf-8 -*-
'''
===================================================
@Time    : 2019/11/28  12:15 上午
@Author  : Simon Chen Xumin
@IDE     : PyCharm
===================================================
'''
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

continous_features = 13


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
        self.feature = []
        self.label = []
        self.train = train

        if self.train:
            data = pd.read_csv(filepath)
            self.train_data = data.iloc[:, :-2].values
            self.target = data.iloc[:, -2].values

        else:
            data = pd.read_csv(filepath)
            self.test_data = data.iloc[:, :-2].values

    def __getitem__(self, index):
        if self.train:
            data_index, target_index = self.train_data[index, :], self.target[index]

            Xi_coutinous = np.zeros_like(data_index[:continous_features])
            Xi_categorial = data_index[continous_features:]
            Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)

            Xv_categorial = np.ones_like(data_index[continous_features:])
            Xv_coutinous = data_index[:continous_features]
            Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))
            return Xi, Xv, target_index
        else:
            data_index = self.train_data[index, :]

            Xi_coutinous = np.zeros_like(data_index[:continous_features])
            Xi_categorial = data_index[continous_features:]
            Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)

            Xv_categorial = np.ones_like(data_index[continous_features:])
            Xv_coutinous = data_index[:continous_features]
            Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
