
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
        self.n_user = len(torch.unique(data[:,0]))
        self.n_movie = len(torch.unique(data[:,1]))
        self.simi_mat = self.cal_simi_mat(data)
        # self.score=self.predict_score(data,1,5)


    def forward(self,user_id,movie_id,):
        print("Dai zhen da shuai b")


    def cal_similarity(self, user_id_1, user_id_2, data):
        # 把目标用户i和j的矩阵和看过的电影放到新的矩阵中
        # users_real1=torch.from_numpy(np.empty((1,2)))
        users_real1=[]
        users_real2=[]

        for x  in range(self.n_user):
            if data[x][0] == user_id_1:
                users_real1.append(data[x][1:])
            if data[x][0] == user_id_2:
                users_real2.append(data[x][1:])

        i_target=[]
        j_target=[]
        if len(users_real1)>len(users_real2):
            for i in users_real2:
                for j in users_real1:
                    if i[0]==j[0]:
                        i_target.append(i[1])
                        j_target.append(j[1])
        else:
            for i in users_real1:
                for j in users_real2:
                    if i[0] == j[0]:
                        i_target.append(i[1])
                        j_target.append(j[1])

        if len(i_target)==1:
            i_target.append(torch.tensor(1.0))
            j_target.append(torch.tensor(1.0))

        # 电影评分->
        if len(i_target)==0:
            similarity=-1
        else:
            if self.criterion == 'pearson':
                result = np.corrcoef(i_target, j_target)
                similarity = result[0,1]
                # torch.c
            else:
                if np.std(users_real1) > 1e-3:  # 方差过大，表明用户间评价尺度差别大需要进行调整
                    users_real1 = users_real1 - users_real1.mean()
                if np.std(users_real2) > 1e-3:
                    users_real2 = users_real2 - users_real2.mean()
                similarity = (users_real1 @ users_real2) / np.linalg.norm(users_real1, 2) / np.linalg.norm(users_real2,2)

        # 如果使用余弦相似度不ok,就换成pearson相关系数来计算相似度
        #使用求出来相似度求方差
        # np.var(similarity[0, 1:], )
        # array_similiarity=similarity[0,1:].A
        # index = np.argmax(array_similiarity)
        # new_array = np.delete(array_similiarity, index)
        # sumb
        return similarity

    def cal_simi_mat(self, data):
        simi_mat = torch.zeros((self.n_user, self.n_user))

        for i in range(self.n_user):
            for j in range(i + 1, self.n_user):
                simi_mat[i, j] = self.cal_similarity(i+1, j+1, data)
                simi_mat[j, i] = simi_mat[i, j]
                # print(i,j,simi_mat[i, j])
        return simi_mat

    # sil_matrix shoule be model.simi_mat
    def predict_score(self,data,target_user,target_movie,target_itemk=6):
        similarity_target_user=self.simi_mat[target_user,:]
        new_array = similarity_target_user
        # 这是空索引数组
        index = np.empty([target_itemk+1], dtype=int)
        #获取最大值处的所在位置,放到数组index里面,再把当前最大的位置删掉,方便下一个循环使用argmax
        for i in range(len(new_array)):
            index[i]=np.argmax(new_array)
            new_array[index[i]]=-1
            if(i>=target_itemk):
                 break

        sum=0
        pre_score=0
        score=0
        user_arr = np.array(data[:, 0])
        movie_arr = np.array(data[:, 1])
        needed_movieindex = torch.flatten(torch.tensor(list(movie_arr[np.where(user_arr == target_movie)])))
        for x in range(index.size):
            # if(index[j]!=0):

             # userindex获取的是相似度最近的userid在data里面的索引，
             # movieindex获取的是所有目标电影在data里面的索引
             # 遍历userid 每次选择第一个与movieid最近的一个，然后属于目前userid的score=data[needed_userindex,3]
             needed_userindex=user_arr[np.where(user_arr==index[x])]
             for i in range(6):
                 for j in range(needed_movieindex.size):
                    if(needed_userindex[i]>=needed_movieindex[j]):
                        real_index=needed_userindex[i]
                        score=data[real_index,3]
                        sum=score+sum
             # print(user_arr[np.where(user_arr==index[j])])
             # print(movie_arr[np.where(user_arr==target_movie)])
             pre_score=sum/target_itemk
             print(pre_score)
        return pre_score




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
