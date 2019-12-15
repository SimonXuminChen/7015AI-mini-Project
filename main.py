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

from tools.preprocessing import PreProcessData
from models.DeepFM import DeepFM

from models.fm import FM
from models.CF_Knearest import CF_knearest
# 1,3671,3.0,1260759117
# 2，31
# 3,588
np.seterr(divide='ignore',invalid='ignore')

CF_TRAIN_DATA_NUMBER = 60000
FM_TRAIN_DATA_NUMBER = 10000
# load data
CF_train_data = PreProcessData("./data/ratings_small.csv",one_hot=False)
FM_train_data = PreProcessData("./data/ratings_small.csv",one_hot=True,is_print=True)

raw_dataset = CF_train_data.feature
dataset = raw_dataset[:CF_TRAIN_DATA_NUMBER,:]

"""
collaborative filtering model
"""

CF_start_time=time.time()
model = CF_knearest(dataset)
print(model.predict_score(dataset,1,3671))
CF_end_time=time.time()
print("the end of training CF, and the time consumption: %d" % (CF_end_time-CF_start_time))

# val_data = PreProcessData("./data/ratings_small.csv", train=True)
# loader_val = DataLoader(train_data, batch_size=50, sampler=sampler.SubsetRandomSampler(range(TRAIN_ROUND, 100000)))




feature_size = FM_train_data.feature_size
print("feature_size is " + str(feature_size))
loader_train = DataLoader(FM_train_data, batch_size=50,
                          sampler=sampler.SubsetRandomSampler(range(FM_TRAIN_DATA_NUMBER)))


"""
FM
"""

FM_start_time=time.time()
FM_model = FM(feature_sizes=feature_size)
print("Now, lets train the model")
FM_model.fit(loader_train,epochs=50)
FM_end_time=time.time()
print("the end of training FM, time consume: %d" % (FM_end_time-FM_start_time))


"""
DeepFM
"""
deepFM_start_time=time.time()
deepFM_model = DeepFM(feature_sizes=feature_size)
print("Now, lets train the model")
deepFM_model.fit(loader_train,epochs=50)
deepFM_end_time=time.time()
print("the end of training deefFM, and the time consumption: %d" % (deepFM_end_time-deepFM_start_time))
