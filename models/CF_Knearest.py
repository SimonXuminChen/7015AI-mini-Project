#  -*- coding: utf-8 -*-
'''
===================================================
@Time    : 2019/12/10  4:56 下午
@Author  : Dai
@IDE     : PyCharm
===================================================
'''

import torch
import torch.nn as nn
import numpy as np


class CF_knearest(nn.Module):
    """ CF&TopN"""

    def __init__(self, data, criterion='pearson'):
        super(CF_knearest, self).__init__()
        self.criterion = criterion
        self.simi_mat = None
        self.n_user = len(torch.unique(data[:, 0]))
        self.n_movie = len(torch.unique(data[:, 1]))
        self.simi_mat = self.cal_simi_mat(data)

    def forward(self, user_id, movie_id, ):
        print("Start")

    #Calculate the similarity between user_id_1 and user_id_2
    def cal_similarity(self, user_id_1, user_id_2, data):
        users_real1 = []
        users_real2 = []

        for x in range(self.n_user):
            if data[x][0] == user_id_1:
                users_real1.append(data[x][1:])
            if data[x][0] == user_id_2:
                users_real2.append(data[x][1:])

        i_target = []
        j_target = []
        if len(users_real1) > len(users_real2):
            for i in users_real2:
                for j in users_real1:
                    if i[0] == j[0]:
                        i_target.append(i[1])
                        j_target.append(j[1])
        else:
            for i in users_real1:
                for j in users_real2:
                    if i[0] == j[0]:
                        i_target.append(i[1])
                        j_target.append(j[1])

        if len(i_target) == 1:
            i_target.append(torch.tensor(1.0))
            j_target.append(torch.tensor(1.0))

        if len(i_target) == 0:
            similarity = -1
        else:
            if self.criterion == 'pearson':
                result = np.corrcoef(i_target, j_target)
                similarity = result[0, 1]
            else:
                if np.std(users_real1) > 1e-3:
                    users_real1 = users_real1 - users_real1.mean()
                if np.std(users_real2) > 1e-3:
                    users_real2 = users_real2 - users_real2.mean()
                similarity = (users_real1 @ users_real2) / np.linalg.norm(users_real1, 2) / np.linalg.norm(users_real2,2)
        return similarity

    #Construct the matrix of similarity
    def cal_simi_mat(self, data):
        simi_mat = torch.zeros((self.n_user, self.n_user))

        for i in range(self.n_user):
            for j in range(i + 1, self.n_user):
                simi_mat[i, j] = self.cal_similarity(i + 1, j + 1, data)
                simi_mat[j, i] = simi_mat[i, j]
        return simi_mat

    #predict score of target_user
    def predict_score(self, data, target_user, target_movie, target_itemk=6):
        similarity_target_user = self.simi_mat[target_user, :]
        new_array = similarity_target_user
        index = np.zeros([target_itemk + 1], dtype=int)
        for i in range(len(new_array)):
            if (i > target_itemk):
                break
            index[i] = np.argmax(new_array)
            new_array[index[i]] = -1


        user_arr = data[:, 0]
        movie_arr = data[:, 1]
        needed_movieindex = torch.flatten(torch.tensor(list(np.where(movie_arr == target_movie))))
        user_matrix = []
        for x in index:
            needed_userindex = torch.flatten(torch.tensor(list(np.where(user_arr == x))))
            user_matrix.append(needed_userindex)

        sum = 0
        pre_score = 0
        score = 0
        n=0
        for i in range(len(user_matrix)):
            if (len(user_matrix[i]) == 0):
                continue
            for j in range(len(needed_movieindex)):
                if (user_matrix[i][0] <= needed_movieindex[j] and needed_movieindex[j] <= user_matrix[i][-1]):
                    real_index = needed_movieindex[j]
                    score = data[real_index, 2]
                    n+=1
                    sum += score

        if(n==0):
            pre_score=sum
        else:
            pre_score = sum / n
        print("pre_score: %.4f"%pre_score)
        return pre_score

