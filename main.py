#  -*- coding: utf-8 -*-
'''
===================================================
@Time    : 2019/12/6  5:30 下午
@Author  : Simon Chen Xumin
@IDE     : PyCharm
===================================================
'''
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from tools.preprocessing import PreProcessData
from models.DeepFM import DeepFM

TRAIN_ROUND = 500

# load data
train_data = PreProcessData("./data/ratings_small.csv",train=True)
loader_train = DataLoader(train_data, batch_size=100,
                          sampler=sampler.SubsetRandomSampler(range(TRAIN_ROUND)))
val_data = PreProcessData("./data/ratings_small.csv",train=True)
loader_val = DataLoader(train_data, batch_size=100,
                          sampler=sampler.SubsetRandomSampler(range(TRAIN_ROUND,10000)))


feature_size = train_data.feature_size
print("feature_size is "+str(feature_size))

model = DeepFM(feature_sizes=feature_size)

print("Now, lets train the model")

model.fit(loader_train,loader_val,epochs=50)