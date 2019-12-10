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

from tools.preprocessing import PreProcessData,load_data
from models.DeepFM import DeepFM

from models.fm import FM

TRAIN_ROUND = 10000

# load data
# train_data = PreProcessData("./data/ratings_small.csv")
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

