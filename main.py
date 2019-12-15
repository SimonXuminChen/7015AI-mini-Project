#  -*- coding: utf-8 -*-
'''
===================================================
@Time    : 2019/12/6  5:30 下午
@Author  : Simon Chen Xumin
@IDE     : PyCharm
===================================================
'''
import time
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np

from tools.preprocessing import PreProcessData,load_data
from models.DeepFM import DeepFM

from models.fm import FM
from models.CF_Knearest import CF_knearest
# 1,3671,3.0,1260759117
# 2，31
# 3,588
np.seterr(divide='ignore',invalid='ignore')

TRAIN_ROUND = 25000
# load data
train_data = PreProcessData("./data/ratings_small.csv",one_hot=False)

raw_dataset = train_data.feature
dataset = raw_dataset[:TRAIN_ROUND,:]

#CF
CF_start_time=time.time()
model = CF_knearest(dataset)
print(model.predict_score(dataset,1,3671))
CF_end_time=time.time()
print("the end of training deefFM, time consume: %d" % (CF_end_time-CF_start_time))

# print(model.predict_score(dataset,1,5)) 会输出[3 0 0 0 0 0 0]
# a=[2.0,3.0]
# b=[2.0,4.5]
# c=np.corrcoef(a,b)
# print(c)
# loader_train = DataLoader(train_data, batch_size=50,
#                           sampler=sampler.SubsetRandomSampler(range(TRAIN_ROUND)))
# val_data = PreProcessData("./data/ratings_small.csv", train=True)
# loader_val = DataLoader(train_data, batch_size=50, sampler=sampler.SubsetRandomSampler(range(TRAIN_ROUND, 100000)))
#
# feature_size = train_data.feature_size
# print("feature_size is " + str(feature_size))
#
#
# # deepFM
# deepFM_start_time=time.time()
# deepFM_model = DeepFM(feature_sizes=feature_size)
# print("Now, lets train the model")
# deepFM_model.fit(loader_train,epochs=50)
# deepFM_end_time=time.time()
# print("the end of training deefFM, time consume: %d" % (deepFM_end_time-deepFM_start_time))

# # FM
# FM_start_time=time.time()
# FM_model = FM(feature_sizes=feature_size)
# print("Now, lets train the model")
# FM_model.fit(loader_train,epochs=50)
# FM_end_time=time.time()
# print("the end of training FM, time consume: %d" % (FM_end_time-FM_start_time))

