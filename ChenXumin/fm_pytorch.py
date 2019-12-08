#  -*- coding: utf-8 -*-
'''
===================================================
@Time    : 2019/11/29  2:11 下午
@Author  : Simon Chen Xumin
@IDE     : PyCharm
===================================================
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
sys.path.append("..")
from tools import preprocessing


# np.seterr(divide='ignore',invalid='ignore')

class FM_Layer(nn.Module):
    def __init__(self,n=10,k=5):
        super(FM_Layer, self).__init__()
        self.n = n
        self.k=k
        self.linear = nn.Linear(self.n,1)
        self.v=nn.Parameter(torch.randn(self.n,self.k))

    def fm_layer(self,x):
        linear_part = self.linear(x)
        interaction_1 = torch.mm(x,self.v)
        interaction_1 = torch.pow(interaction_1,2)
        interaction_2 = torch.mm(torch.pow(x,2),torch.pow(self.v,2))
        output = linear_part + torch.sum(0.5*interaction_2-interaction_1)
        return output

    def forward(self, x):
        return self.fm_layer(x)




feature, labels = preprocessing.load_data("./ratings.csv")
dataMatrix = torch.mm(Variable(torch.from_numpy(feature)))
fm = FM_Layer
x = torch.randn(1,10)
output = fm(x)