import torch
import torch.nn as nn

class TopK(nn.Module):
    def __init__(self,distance,k):
        super(TopK, self).__init__()
        self.distance=distance
        self.k=k

    def forward(self):
        topk=torch.topk(self.distance,5)
        return topk