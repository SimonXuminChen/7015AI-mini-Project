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

import torch
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable

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
        label.append(float(lines[-2])*2)
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
            logging.warning( "There is a null, so this might be processed")
            flag=False
            break
        else:
            new_list.append(list[i])

    if (not list[2].isalpha()) and ((float(list[2]) < min) or (float(list[2]) > max)):
        # there is a outlier value,so we have to delete this line
        logging.warning("There is outlier, so this might be processed")
        flag=False

    if flag is True:
        new_list.append(list[2])
        return new_list
    else:
        return []


class movie_Dataset(Dataset):
    def __init__(self,filename):
        dataset = np.loadtxt(filename,delimiter=',',skiprows=1,dtype=np.float32)
        self.feature = []
        self.label = []
        for line in dataset.readlines():
            feature_tmp = []
            lines = outlier_detect(line.strip().split(','))
            if len(lines) == 0:
                continue
            for x in range(len(lines) - 2):
                feature_tmp.append(float(lines[x]))
            self.label.append(float(lines[-2]) * 2)
            self.feature.append(feature_tmp)
        self.feature = torch.from_numpy(self.feature)
        self.label = torch.from_numpy(self.label)

        self.len=dataset.shape[0]

    def __getitem__(self, index):
        return self.feature[index],self.label[index]

    def __len__(self):
        return self.len