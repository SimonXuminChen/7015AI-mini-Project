
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
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class CF_knearest(nn.Module):
    """ 基于user的K近邻协同过滤推荐算法"""
    def __init__(self,data, criterion='pearson'):
        super(CF_knearest, self).__init__()
        self.criterion = criterion
        self.simi_mat = None
        self.n_user = data[:,0]
        self.n_movie = data[:,1]
        self.simi_mat = self.cal_simi_mat(data)

        return

    def forward(self,user_id,movie_id,):
        self.cal_similarity()


    def cal_similarity(self, i, j, data):
        # 把目标用户i和j的矩阵和看过的电影放到新的矩阵中
        # users_real1=torch.from_numpy(np.empty((1,2)))
        users_real1=[]
        users_real2=[]

        for x  in range(len(data)):
            if data[x][0] == i:
                users_real1.append(data[x][1])
            if data[x][0] == j:
                np.append(users_real2, data[x][1], axis=None)

        if len(users_real1) == 0:
            if len(users_real2)==0:
                similarity = 0
        else:
            v1 = users_real1[0,:]
            v2 = users_real2[0,:]
            if self.criterion == 'cosine':
                if np.std(v1) > 1e-3:  # 方差过大，表明用户间评价尺度差别大需要进行调整
                    v1 = v1 - v1.mean()
                if np.std(v2) > 1e-3:
                    v2 = v2 - v2.mean()
                similarity = (v1 @ v2) / np.linalg.norm(v1, 2) / np.linalg.norm(v2, 2)
            # 如果使用余弦相似度不ok,就换成pearson相关系数来计算相似度
            elif self.criterion == 'pearson':
                similarity = np.corrcoef(v1, v2)[0, 1]
            else:
                raise ValueError('the method is not supported now')

        return similarity

    def cal_simi_mat(self, data):
        simi_mat = np.ones((self.n_user, self.n_user))

        for i in range(self.n_user):
            for j in range(i + 1, self.n_user):
                simi_mat[i, j] = self.cal_similarity(i, j, data)
                simi_mat[j, i] = simi_mat[i, j]
        return simi_mat

    def predict_score(self,data,target_user,target_movie,sil_matrix,target_itemk=6):

        # 获取目标用户在相似度矩阵重的所有行，也就是目标用户和其他所有用户之间的  相似度
        similarity_target_user=sil_matrix[target_user,:]
        # topk=similarity_target_user.sort_values('similarity',ascending=False)[1:target_itemk+1]
        new_array=similarity_target_user.A
        # 把similarity_target_user换成array数组，再用np.argmax
        # 这是空索引数组
        index = np.empty([target_itemk], dtype=int)

        #获取最大值处的所在位置,放到数组index里面,再把当前最大的位置删掉,方便下一个循环使用argmax
        for i in range(new_array.size):
            index=np.argmax(new_array)
            new_array=np.delete(new_array,index)
            if(index.size>=target_itemk):
                return index

        # 对 data[index,target_movie] 位置处的求平均数
        sum=0
        score=0
        for j in range(index.size):
            sum=data[index[j],target_movie]+sum
            while (j>index.size):
                score=sum/target_itemk
            return score



    # def fit(self,real_score,predict_score):
    #
    #     # loss_function= F.mse_loss
    #     #
    #     # input = torch.autograd.Variable(real_score)
    #     # target = torch.autograd.Variable(predict_score)
    #     # loss = loss_function(input.float(), target.float())
    #     # return loss

        # index=topk.userid
        # single_score=lable[userid,target_movie]
        # score=sum(single_score)

    # def predict(self,simi_mat,num_user):
    #     for i in simi_mat:
    #         if(i>0.95):

    #       相似度高于0.95的几个用户，使用
    # #没理解 forward:
    # def cal_prediction(self, user_row, movie_ind):
    #     # 计算预推荐电影i对目标活跃用户u的吸引力
    #     purchase_movie_inds = np.where(user_row > 0)[0]
    #     rates = user_row[purchase_movie_inds]
    #     simi = self.simi_mat[movie_ind][purchase_movie_inds]
    #     return np.sum(rates * simi) / np.linalg.norm(simi, 1)

    # def cal_recommendation(self, user_ind, data):
    #     # 计算目标用户的最具吸引力的k个物品list
    #     item_prediction = defaultdict(float)
    #     user_row = data[user_ind]
    #     un_purchase_item_inds = np.where(user_row == 0)[0]
    #     for item_ind in un_purchase_item_inds:
    #         item_prediction[item_ind] = self.cal_prediction(user_row, item_ind)
    #     res = sorted(item_prediction, key=item_prediction.get, reverse=True)
    #     return res[:self.k]
