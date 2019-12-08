import torch
import torch.nn.functional as F
import torch.nn as nn

#movie_1和2代表同类型2个电影的在
class CousinM(nn.Module):
    def __init__(self, a_movie_1, a_movie_2,b_movie_1,b_movie_2):
        super(CousinM, self).__init__()
        self.a_movie_1=a_movie_1.view(a_movie_1.shape[1], -1)
        self.a_movie_2=a_movie_2.view(a_movie_2.shape[1], -1)
        self.b_movie_1=b_movie_1.view(b_movie_1.shape[1], -1)
        self.b_movie_2=b_movie_2.view(b_movie_2.shape[1], -1)

    def forward(self):
        feature1 = F.normalize(self.a_movie_1+self.a_movie_2)  #F.normalize只能处理两维的数据，L2归一化
        feature2 = F.normalize(self.b_movie_1+self.b_movie_2)
        distance = feature1.mm(feature2.t())#计算余弦相似度
        #Or use cosine_similarity(x,y,dim=1)方法
        return distance