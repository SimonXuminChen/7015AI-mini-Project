import torch
import torch.nn as nn

class User_base(nn.Module):
    def __init__(self,index,input):
        '''
        定义模型基础数据，学习率，模型结构
        '''
        super(User_base, self).__init__()
        self.index=index
        self.input=input

    def forward(self):
        '''
        定义图走向（计算逻辑顺序）
        '''

